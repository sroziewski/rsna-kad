<#
.SYNOPSIS
  Drive kaggle_fetch.py on the GPU server from this Windows machine.
  Set $env:SERVER_IP (server host) and $env:KEY_PATH (SSH private key file; $env:SERVER_KEYS also works),
  $env:REMOTE_DIR is the project dir on the server. Or pass -Server / -KeyPath / -RemoteDir.

.EXAMPLE
  .\scripts\remote.ps1 deploy      # copy the fetcher to the server
  .\scripts\remote.ps1 token       # prompt for the Kaggle API token, store it on the server (chmod 600)
  .\scripts\remote.ps1 preflight   # check token, rules acceptance, archive size, disk space
  .\scripts\remote.ps1 start       # run the full fetch in a detached screen session "fetcher"
  .\scripts\remote.ps1 status      # is it running + last log lines + disk usage
  .\scripts\remote.ps1 logs        # follow the current log (Ctrl+C to leave; the fetch keeps running)
  .\scripts\remote.ps1 stop        # interrupt the fetch (it can be resumed with start)
  .\scripts\remote.ps1 sync        # push the project tree (scripts, src, tests, configs, notebooks, analysis) to the server
  .\scripts\remote.ps1 drift       # only the md5/mtime check sync runs first; exit 1 when it would refuse
  .\scripts\remote.ps1 pull        # fetch data/ and analysis/out (small artefacts) back from the server
  .\scripts\remote.ps1 run 'nvidia-smi'   # run any command on the server, in the project dir
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateSet('deploy', 'token', 'preflight', 'start', 'status', 'logs', 'stop', 'test', 'sync', 'drift', 'pull', 'run')]
    [string]$Action,

    # For 'run': the remote command.
    [Parameter(Position = 1)]
    [string]$Command,

    # 'sync' only: also push the local data/ tree, overwriting the server's copy.
    [switch]$PushData,

    # 'sync' only: push even when the drift check finds server files the local copy did not originate.
    [switch]$Force,

    [string]$Server = $env:SERVER_IP,
    [string]$User = 'sroziewski',
    [string]$KeyPath = $(if ($env:KEY_PATH) { $env:KEY_PATH } else { $env:SERVER_KEYS }),
    [string]$RemoteDir = $env:REMOTE_DIR,
    [string]$DataDir = '/raid/sroziewski/data/rsna_kad',
    [string]$Python = '/raid/sroziewski/sandbox/bin/python',
    [string]$Session = 'fetcher',
    # Legacy credentials file, as an alternative to pasting a token.
    [string]$KaggleJson
)

$ErrorActionPreference = 'Stop'
if (-not $Server) { throw 'No server: set $env:SERVER_IP or pass -Server <host>.' }
if (-not $RemoteDir) { throw 'No remote project dir: set $env:REMOTE_DIR or pass -RemoteDir <path>.' }
if (-not $KeyPath) { throw 'No SSH key: set $env:KEY_PATH (or $env:SERVER_KEYS) or pass -KeyPath <private key file>.' }
$repo = Split-Path $PSScriptRoot -Parent
$target = "$User@$Server"
$tempKey = $null

function Get-UsableKey {
    # OpenSSH refuses a private key that anyone but the owner, SYSTEM or
    # Administrators can read. Rather than rewriting the ACL of a key shared
    # with MobaXterm, work from a locked-down throwaway copy.
    $me = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $trusted = @($me, 'S-1-5-18', 'S-1-5-32-544')
    $others = (Get-Acl $KeyPath).Access | Where-Object {
        $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -notin $trusted
    }
    if (-not $others) { return $KeyPath }

    # Copies orphaned by a killed console; a day is far longer than any run.
    Get-ChildItem (Join-Path $env:TEMP 'kad-key-*') -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-1) } | Remove-Item -Force
    $copy = Join-Path $env:TEMP ("kad-key-" + [guid]::NewGuid().ToString('N'))
    Copy-Item $KeyPath $copy
    icacls $copy /inheritance:r /grant:r "*${me}:(F)" | Out-Null
    $script:tempKey = $copy
    return $copy
}

function Invoke-Remote([string]$Command, [switch]$Tty) {
    $sshArgs = @('-i', $key, '-o', 'StrictHostKeyChecking=accept-new')
    if ($Tty) { $sshArgs += '-t' } else { $sshArgs += @('-o', 'BatchMode=yes') }
    $Command = $Command -replace "`r", ''
    if ($MyInvocation.ExpectingInput) { $input | ssh @sshArgs $target $Command } else { ssh @sshArgs $target $Command }
    if ($LASTEXITCODE -ne 0) { throw "remote command failed (exit $LASTEXITCODE)" }
}

$syncManifest = Join-Path $repo '.sync-manifest.tsv'

function Get-SyncFiles([string[]]$Items) {
    # Exactly what the tar in 'sync' sends (same excludes): repo-relative path -> FileInfo.
    $files = @{}
    foreach ($item in $Items) {
        foreach ($f in Get-ChildItem (Join-Path $repo $item) -Recurse -File -Force) {
            $rel = [IO.Path]::GetRelativePath($repo, $f.FullName).Replace('\', '/')
            if ($rel -match '(^|/)__pycache__(/|$)|\.pyc$|^analysis/out(/|$)|\.egg-info(/|$)') { continue }
            $files[$rel] = $f
        }
    }
    return $files
}

function Find-SyncDrift([hashtable]$Files) {
    # A differing file may be pushed only when the difference started locally. It did not when the
    # server copy is newer than the local one, or when the server copy no longer has the md5 the
    # last sync left there (.sync-manifest.tsv). The real case: a stale local notebook that sync
    # would have pushed over the copy the live kernel runs.
    $manifest = @{}
    if (Test-Path $syncManifest) {
        foreach ($line in Get-Content $syncManifest) { $md5, $rel = $line -split "`t", 2; $manifest[$rel] = $md5 }
    }
    $list = ($Files.Keys | Sort-Object) -join "`n"
    $probe = "cd $RemoteDir || exit 1`nwhile IFS= read -r f; do f=`${f%`$'\r'}; " +
        "[ -f `"`$f`" ] && printf '%s %s %s\n' `"`$(md5sum < `"`$f`" | cut -c1-32)`" `"`$(stat -c %Y `"`$f`")`" `"`$f`"; " +
        "done <<'__KAD_LIST__'`n$list`n__KAD_LIST__`nexit 0`n"
    $drift = [Collections.Generic.List[string]]::new()
    $changed = 0
    foreach ($line in ($probe | Invoke-Remote 'bash -s')) {
        $md5, $mtime, $rel = "$line" -split ' ', 3
        $f = $Files[$rel]
        if (-not $f) { continue }
        if ((Get-FileHash -Algorithm MD5 $f.FullName).Hash.ToLowerInvariant() -eq $md5) { continue }
        $changed++
        $localTime = [DateTimeOffset]::new($f.LastWriteTimeUtc).ToUnixTimeSeconds()
        if ([long]$mtime -gt $localTime) {
            $drift.Add("$rel  (server copy is newer: server $([DateTimeOffset]::FromUnixTimeSeconds([long]$mtime).ToString('u')), local $($f.LastWriteTimeUtc.ToString('u')))")
        } elseif ($manifest.ContainsKey($rel) -and $manifest[$rel] -ne $md5) {
            $drift.Add("$rel  (server copy changed since the last sync)")
        }
    }
    return [pscustomobject]@{ Drift = $drift; Changed = $changed }
}

function Save-SyncManifest([hashtable]$Files) {
    $lines = foreach ($rel in ($Files.Keys | Sort-Object)) {
        "$((Get-FileHash -Algorithm MD5 $Files[$rel].FullName).Hash.ToLowerInvariant())`t$rel"
    }
    Set-Content -Path $syncManifest -Value $lines -Encoding utf8NoBOM
}

$fetch = "$Python $RemoteDir/scripts/kaggle_fetch.py --data-dir $DataDir --config-dir $RemoteDir/.kaggle"

try {
    $key = Get-UsableKey

    switch ($Action) {
        'deploy' {
            Invoke-Remote "mkdir -p $RemoteDir/scripts $RemoteDir/tests && mkdir -p -m 700 $RemoteDir/.kaggle"
            $scpArgs = @('-i', $key, '-o', 'BatchMode=yes')
            foreach ($file in 'scripts/kaggle_fetch.py', 'tests/test_kaggle_fetch.py') {
                scp @scpArgs (Join-Path $repo $file) "${target}:$RemoteDir/$file"
                if ($LASTEXITCODE -ne 0) { throw "scp of $file failed" }
            }
            Write-Host "deployed to ${target}:$RemoteDir"
        }
        'token' {
            if ($KaggleJson) {
                $secret = Get-Content $KaggleJson -Raw
                $file = 'kaggle.json'
            } else {
                Write-Host 'Create one at https://www.kaggle.com/settings/api ("Generate New Token").'
                $secure = Read-Host 'Kaggle API token' -AsSecureString
                $secret = [Net.NetworkCredential]::new('', $secure).Password
                $file = 'access_token'
            }
            if (-not $secret.Trim()) { throw 'empty token' }
            # Sent over stdin so the secret never shows up in a process list
            # or shell history on either machine.
            # PowerShell appends CRLF when piping; the kaggle CLI puts the file's bytes verbatim into
            # the Authorization header, so strip line endings server-side.
            $secret | Invoke-Remote "umask 077 && mkdir -p $RemoteDir/.kaggle && tr -d '\r\n' > $RemoteDir/.kaggle/$file && chmod 600 $RemoteDir/.kaggle/$file"
            Write-Host "stored as $RemoteDir/.kaggle/$file (mode 600)"
        }
        'preflight' { Invoke-Remote "$fetch preflight" }
        'start' {
            Invoke-Remote @"
if screen -ls | grep -q '[.]$Session[[:space:]]'; then echo 'screen session $Session is already running'; exit 1; fi
screen -dmS $Session $fetch all
sleep 3
if screen -ls | grep '[.]$Session[[:space:]]'; then
  echo 'started. Attach on the server with: screen -r $Session   (detach: Ctrl+A, D)'
else
  echo 'the fetcher exited right away; its log says:'
  ls -t $DataDir/.download/logs/*.log 2>/dev/null | head -1 | xargs -r tail -n 5
  exit 1
fi
"@
        }
        'status' {
            Invoke-Remote @"
screen -ls | grep '[.]$Session' || echo 'screen session ${Session} is not running'
echo '--- last log lines'; ls -t $DataDir/.download/logs/*.log 2>/dev/null | head -1 | xargs -r tail -n 8
echo '--- disk'; ls -lh $DataDir/.download/ 2>/dev/null | grep -E 'zip|COMPLETE'; df -h $DataDir | tail -1
"@
        }
        'logs' { Invoke-Remote "ls -t $DataDir/.download/logs/*.log 2>/dev/null | head -1 | xargs -r tail -n 30 -f" -Tty }
        'stop' {
            # Ctrl+C lets the fetcher finish the file in flight and exit cleanly.
            Invoke-Remote "screen -S $Session -X stuff `$'\003' && echo 'interrupt sent; resume later with: remote.ps1 start'"
        }
        { $_ -in 'sync', 'drift' } {
            # data/ is PRODUCED on the server (label tables, targets, folds) and only ever travels
            # back via 'pull'; pushing it would overwrite results that training chains are reading.
            $items = @('pyproject.toml', 'scripts', 'src', 'tests', 'configs', 'notebooks', 'analysis') |
                Where-Object { Test-Path (Join-Path $repo $_) }
            if ($PushData) { $items += 'data' }
            $files = Get-SyncFiles $items
            $check = Find-SyncDrift $files
            Write-Host "$($files.Count) files checked, $($check.Changed) differ from the server, $($check.Drift.Count) not originated locally"
            if ($check.Drift.Count) {
                Write-Host "server files the local copy did not originate (pull them first, or -Force to overwrite):"
                $check.Drift | ForEach-Object { Write-Host "  $_" }
                if ($Action -eq 'drift' -or -not $Force) { exit 1 }
            }
            if ($Action -eq 'drift') { break }
            # Windows ships bsdtar; streaming a tarball avoids one scp per file.
            $sshArgs = @('-i', $key, '-o', 'BatchMode=yes', $target, "mkdir -p $RemoteDir && tar -xf - -C $RemoteDir")
            tar -cf - -C $repo --exclude '__pycache__' --exclude '*.pyc' --exclude 'analysis/out' --exclude '*.egg-info' @items | ssh @sshArgs
            if ($LASTEXITCODE -ne 0) { throw 'sync failed' }
            Save-SyncManifest $files
            Write-Host "synced $($items -join ', ') to ${target}:$RemoteDir"
        }
        'pull' {
            # Small artefacts produced on the server (label tables, folds, metrics) come back into the repo.
            $sshArgs = @('-i', $key, '-o', 'BatchMode=yes', $target,
                "cd $RemoteDir && tar -cf - --exclude '*.npy' --exclude 'runs' data analysis/out 2>/dev/null")
            ssh @sshArgs | tar -xf - -C $repo
            if ($LASTEXITCODE -ne 0) { throw 'pull failed' }
            Write-Host "pulled data/ and analysis/out from ${target}:$RemoteDir"
        }
        'run' {
            if (-not $Command) { throw "usage: remote.ps1 run '<command>'" }
            Invoke-Remote "cd $RemoteDir && $Command" -Tty
        }
        'test' {
            Invoke-Remote "cd $RemoteDir && mkdir -p .test_tmp && TMPDIR=$RemoteDir/.test_tmp PYTHONDONTWRITEBYTECODE=1 $Python -m pytest tests -q -p no:cacheprovider; rc=`$?; rm -rf $RemoteDir/.test_tmp; exit `$rc"
        }
    }
}
finally {
    if ($tempKey -and (Test-Path $tempKey)) { Remove-Item $tempKey -Force }
}
