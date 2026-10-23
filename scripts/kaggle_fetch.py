#!/usr/bin/env python3
"""Fetch a Kaggle competition archive onto a space-constrained server.

Stdlib only, so it runs on any Python >= 3.9 without installing anything.

Phases (each one is resumable - re-run the same command after any failure):
  preflight  check credentials, competition access, archive size, disk space
  download   ranged, resumable download of the competition archive
  extract    parallel extraction; can give archive space back to the
             filesystem while it goes, so archive + extracted data never have
             to fit on disk together
  verify     check the extracted tree against the manifest
  all        everything above, in order

Credentials are looked up in: $KAGGLE_API_TOKEN, <config-dir>/access_token,
$KAGGLE_USERNAME + $KAGGLE_KEY, <config-dir>/kaggle.json. The home directory
is never touched.
"""
from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures
import ctypes
import ctypes.util
import errno
import fcntl
import http.client
import itertools
import json
import logging
import os
import shutil
import signal
import stat
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

log = logging.getLogger("kaggle_fetch")

DEFAULT_COMPETITION = "rsna-knee-abnormality-detection"
DEFAULT_DATA_DIR = "/raid/sroziewski/data/rsna_kad"

# The RPC route is what the official kagglesdk calls; the REST route is the
# older public API. Both answered 401 (i.e. exist) when probed on 2026-09-17.
API_RPC_BASE = "https://api.kaggle.com/v1"
API_REST_BASE = "https://www.kaggle.com/api/v1"

USER_AGENT = "rsna-kad-fetch/1.0 (python-urllib)"
HTTP_TIMEOUT = 60
CHUNK = 4 << 20
GIB = 1 << 30
PROGRESS_EVERY = 30.0
TMP_SUFFIX = ".kfpart"

_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_FALLOC_FL_KEEP_SIZE = 0x01
_FALLOC_FL_PUNCH_HOLE = 0x02


class FetchError(Exception):
    """An expected failure whose message tells the user what to do."""


# --------------------------------------------------------------------------- auth


@dataclass(frozen=True)
class Auth:
    header: str
    source: str


def _warn_if_exposed(path: Path) -> None:
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        log.warning("%s is readable by other users; run: chmod 600 %s", path, path)


def load_auth(config_dir: Path, env=os.environ) -> Auth:
    token = env.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        return Auth(f"Bearer {token}", "env KAGGLE_API_TOKEN")

    token_file = config_dir / "access_token"
    if token_file.is_file():
        _warn_if_exposed(token_file)
        token = token_file.read_text().strip()
        if token:
            return Auth(f"Bearer {token}", str(token_file))

    user, key = env.get("KAGGLE_USERNAME", "").strip(), env.get("KAGGLE_KEY", "").strip()
    source = "env KAGGLE_USERNAME/KAGGLE_KEY"
    legacy = config_dir / "kaggle.json"
    if not (user and key) and legacy.is_file():
        _warn_if_exposed(legacy)
        try:
            creds = json.loads(legacy.read_text())
            user, key = creds["username"].strip(), creds["key"].strip()
        except (ValueError, KeyError, AttributeError) as exc:
            raise FetchError(f"{legacy} is not a valid kaggle.json: {exc!r}") from None
        source = str(legacy)
    if user and key:
        basic = base64.b64encode(f"{user}:{key}".encode()).decode()
        return Auth(f"Basic {basic}", source)

    raise FetchError(
        "no Kaggle credentials found. Create a token at https://www.kaggle.com/settings/api "
        f"and store it in {token_file} (chmod 600), or export KAGGLE_API_TOKEN."
    )


# ---------------------------------------------------------------------- kaggle api


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # The archive URL is a signed storage link: it must be requested without
    # the Kaggle Authorization header, so redirects are surfaced, not followed.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_api_opener = urllib.request.build_opener(_NoRedirect)


def _archive_routes(competition: str):
    slug = urllib.parse.quote(competition, safe="")
    return [
        ("POST", f"{API_RPC_BASE}/competitions.CompetitionApiService/DownloadDataFiles",
         {"competitionName": competition}),
        ("GET", f"{API_REST_BASE}/competitions/data/download-all/{slug}", None),
    ]


def _error_text(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read(2000).decode("utf-8", "replace").strip()
    except OSError:
        return ""
    try:
        doc = json.loads(raw)
    except ValueError:
        return "" if raw.lstrip().startswith("<") else raw[:300]
    if isinstance(doc, dict):
        inner = doc.get("error") if isinstance(doc.get("error"), dict) else doc
        return str(inner.get("message") or raw[:300])
    return raw[:300]


def resolve_archive_url(auth: Auth, competition: str) -> str:
    """Return a short-lived direct URL of the whole-competition archive."""
    failures = []
    for method, url, body in _archive_routes(competition):
        headers = {"Authorization": auth.header, "User-Agent": USER_AGENT, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with _api_opener.open(req, timeout=HTTP_TIMEOUT) as resp:
                failures.append(f"{method} {url} -> HTTP {resp.status} without a redirect")
        except urllib.error.HTTPError as exc:
            location = exc.headers.get("Location")
            if exc.code in (301, 302, 303, 307, 308) and location:
                return urllib.parse.urljoin(url, location)
            detail = _error_text(exc)
            if exc.code == 401:
                raise FetchError(
                    f"Kaggle rejected the credentials from {auth.source} (HTTP 401 {detail}). "
                    "Generate a fresh token at https://www.kaggle.com/settings/api"
                ) from None
            if exc.code == 403:
                raise FetchError(
                    f"access denied (HTTP 403 {detail}). Accept the rules first: "
                    f"https://www.kaggle.com/competitions/{competition}/rules"
                ) from None
            if exc.code == 429 or exc.code >= 500:
                raise
            failures.append(f"{method} {url} -> HTTP {exc.code} {detail}")
    raise FetchError("could not obtain the archive URL:\n  " + "\n  ".join(failures))


# ----------------------------------------------------------------------- helpers


def _write_json_atomic(path: Path, doc) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(doc, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _fmt_size(n: float) -> str:
    return f"{n / GIB:.2f} GiB"


def _fmt_eta(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf") or seconds < 0:
        return "?"
    minutes, _ = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class _Progress:
    def __init__(self, label: str, total: int, done: int = 0):
        self.label, self.total = label, total
        self.start_done = self.done = done
        self.started = self.last = time.monotonic()

    def update(self, done: int, extra: str = "", force: bool = False) -> None:
        self.done = done
        now = time.monotonic()
        if not force and now - self.last < PROGRESS_EVERY:
            return
        self.last = now
        rate = (done - self.start_done) / max(now - self.started, 1e-6)
        eta = (self.total - done) / rate if rate > 0 else float("inf")
        pct = 100.0 * done / self.total if self.total else 100.0
        log.info("%s %s / %s (%.1f%%) %.1f MiB/s ETA %s%s", self.label, _fmt_size(done),
                 _fmt_size(self.total), pct, rate / (1 << 20), _fmt_eta(eta), extra)


class Paths:
    def __init__(self, data_dir: Path, competition: str):
        self.data = data_dir
        self.work = data_dir / ".download"
        self.zip = self.work / f"{competition}.zip"
        self.part = self.work / f"{competition}.zip.part"
        self.part_meta = self.work / f"{competition}.zip.part.meta"
        self.reclaim_state = self.work / f"{competition}.zip.reclaim"
        self.manifest = self.work / "manifest.tsv"
        self.complete = self.work / "COMPLETE"
        self.lock = self.work / "lock"
        self.logs = self.work / "logs"


# ---------------------------------------------------------------------- download


def _parse_content_range(value):
    # "bytes 100-999/1000"
    try:
        unit, _, rest = value.partition(" ")
        span, _, total = rest.partition("/")
        first, _, _last = span.partition("-")
        if unit != "bytes":
            return None
        return int(first), (None if total.strip() == "*" else int(total))
    except (AttributeError, ValueError):
        return None


def probe_archive(url: str):
    """Return (total_bytes, etag, ranges_supported) using a 1-byte request."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        parsed = _parse_content_range(resp.headers.get("Content-Range"))
        if resp.status == 206:
            return (parsed[1] if parsed else None), resp.headers.get("ETag"), True
        length = resp.headers.get("Content-Length")
        return (int(length) if length else None), resp.headers.get("ETag"), False


def download(paths: Paths, resolve, min_free: int, max_failures: int = 30) -> None:
    """Download the archive to paths.zip, resuming paths.part if present.

    `resolve` returns a fresh archive URL; it is called again whenever the
    current (signed, expiring) URL stops working.
    """
    if paths.zip.exists():
        log.info("archive already downloaded: %s", paths.zip)
        return

    url = None
    failures = restarts = 0
    progress = None
    while True:
        offset = paths.part.stat().st_size if paths.part.exists() else 0
        meta = _read_json(paths.part_meta) or {}
        if offset and not meta:
            log.warning("partial file without metadata; restarting the download from zero")
            paths.part.unlink()
            offset = 0
        if meta and offset == meta.get("total"):
            break
        if meta and offset > meta.get("total", 0):
            log.warning("partial file is larger than the archive; restarting from zero")
            paths.part.unlink()
            paths.part_meta.unlink()
            continue

        before = offset
        try:
            if url is None:
                url = resolve()
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={offset}-", "User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                etag = resp.headers.get("ETag")
                parsed = _parse_content_range(resp.headers.get("Content-Range"))
                if resp.status == 206 and parsed and parsed[0] == offset and parsed[1]:
                    total = parsed[1]
                elif resp.status == 200 and resp.headers.get("Content-Length"):
                    total = int(resp.headers["Content-Length"])
                    if offset:
                        log.warning("server ignored the Range header; restarting from zero")
                    offset = 0
                else:
                    # Retried like any other hiccup: proxies occasionally answer
                    # with a shape the origin never would.
                    raise urllib.error.URLError(
                        f"unusable response: HTTP {resp.status} "
                        f"Content-Range={resp.headers.get('Content-Range')!r} "
                        f"Content-Length={resp.headers.get('Content-Length')!r}")

                if offset and (meta.get("etag") != etag or meta.get("total") != total):
                    # The archive was republished since the partial file was
                    # started; its bytes belong to a different object.
                    restarts += 1
                    if restarts > 3:
                        raise FetchError(
                            "the remote archive identity (ETag/size) changes on every request, so "
                            "it cannot be resumed; see the README fallback (official kaggle CLI)")
                    log.warning("remote archive changed (etag/size); restarting from zero")
                    paths.part.unlink()
                    paths.part_meta.unlink()
                    continue

                need = total - offset + min_free
                free = _free_bytes(paths.work)
                if free < need:
                    raise FetchError(
                        f"not enough disk space: {_fmt_size(free)} free, need {_fmt_size(need)} "
                        f"(remaining archive + {_fmt_size(min_free)} reserve)")

                _write_json_atomic(paths.part_meta, {"etag": etag, "total": total})
                if progress is None:
                    progress = _Progress("download", total, offset)
                    log.info("downloading %s (resuming at %s)", _fmt_size(total), _fmt_size(offset))

                with open(paths.part, "r+b" if offset else "wb") as out:
                    out.seek(offset)
                    out.truncate()
                    since_check = 0
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        try:
                            out.write(chunk)
                        except OSError as exc:
                            raise FetchError(f"cannot write {paths.part}: {exc}") from None
                        offset += len(chunk)
                        since_check += len(chunk)
                        progress.update(offset)
                        if since_check >= GIB:
                            since_check = 0
                            if _free_bytes(paths.work) < min_free:
                                raise FetchError(
                                    f"free space fell below the {_fmt_size(min_free)} reserve; "
                                    "stopping (re-run to resume)")
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                log.warning("server rejected the resume offset; restarting from zero")
                paths.part.unlink(missing_ok=True)
                paths.part_meta.unlink(missing_ok=True)
            elif exc.code < 500 and exc.code != 429:
                log.info("archive URL no longer valid (HTTP %s); requesting a new one", exc.code)
            else:
                log.warning("server error HTTP %s", exc.code)
            url = None
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            log.warning("transfer interrupted: %r", exc)

        now = paths.part.stat().st_size if paths.part.exists() else 0
        if now > before:
            failures = 0
            continue
        failures += 1
        if failures >= max_failures:
            raise FetchError(f"giving up after {failures} consecutive attempts without progress")
        delay = min(300, 2 ** failures)
        log.info("retrying in %ss (attempt %s/%s)", delay, failures, max_failures)
        time.sleep(delay)

    if not zipfile.is_zipfile(paths.part):
        raise FetchError(f"{paths.part} is complete but is not a zip archive; delete it and retry")
    os.replace(paths.part, paths.zip)
    paths.part_meta.unlink(missing_ok=True)
    paths.reclaim_state.unlink(missing_ok=True)
    log.info("download complete: %s (%s)", paths.zip, _fmt_size(paths.zip.stat().st_size))


# ---------------------------------------------------------------------- extract


_libc = None


def punch_hole(fd: int, offset: int, length: int) -> None:
    global _libc
    if _libc is None:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
        libc.fallocate.restype = ctypes.c_int
        _libc = libc
    libc = _libc
    mode = _FALLOC_FL_PUNCH_HOLE | _FALLOC_FL_KEEP_SIZE
    if libc.fallocate(fd, mode, offset, length) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _require_hole_punching(directory: Path) -> None:
    probe = directory / "punch-probe"
    try:
        probe.write_bytes(b"\0" * 8192)
        fd = os.open(probe, os.O_RDWR)
        try:
            punch_hole(fd, 0, 4096)
        finally:
            os.close(fd)
    except OSError as exc:
        raise FetchError(f"the filesystem under {directory} cannot give archive space back "
                         f"({exc}); free more space and use --reclaim never") from None
    finally:
        probe.unlink(missing_ok=True)


class _Reclaimer:
    """Gives already-extracted archive ranges back to the filesystem."""

    def __init__(self, fd: int, state_path: Path, step: int, limit: int):
        self.fd, self.state_path, self.step, self.limit = fd, state_path, step, limit
        st = os.fstat(fd)
        self.identity = {"ino": st.st_ino, "size": st.st_size}
        state = _read_json(state_path) or {}
        if state and {k: state.get(k) for k in self.identity} != self.identity:
            log.warning("ignoring %s: it describes a different archive file", state_path)
            state = {}
        self.done = state.get("punched_upto", 0)
        if not state:
            # Whatever an earlier plain-mode run extracted becomes trustworthy
            # here instead of being redone.
            self._sync_and_record(0)
        self.synced_before_ns = (_read_json(state_path) or {}).get("synced_before_ns", 0)

    def _sync_and_record(self, upto: int) -> None:
        # Workers keep writing while sync() runs, so only files finished before
        # it *started* are known to be durable. File timestamps come from the
        # kernel's coarse clock, which may lag the wall clock; hence the margin.
        started = time.time_ns() - 2_000_000_000
        os.sync()
        _write_json_atomic(self.state_path,
                           {"punched_upto": upto, "synced_before_ns": started, **self.identity})

    def advance(self, watermark: int) -> None:
        watermark = min(watermark, self.limit)
        if watermark - self.done < self.step:
            return
        # Order matters. The hole destroys the only other copy of the data, so
        # the extracted files must be on disk first (sync), and the recorded
        # mark must never trail the real hole (state before punch), otherwise
        # a resume would try to read zeroed ranges as valid members.
        self._sync_and_record(watermark)
        try:
            punch_hole(self.fd, self.done, watermark - self.done)
        except OSError as exc:
            self._sync_and_record(self.done)
            raise FetchError(f"could not reclaim archive space: {exc}") from None
        log.info("reclaimed archive space up to %s", _fmt_size(watermark))
        self.done = watermark


def _safe_relpath(name: str) -> PurePosixPath:
    rel = PurePosixPath(name)
    if not name or rel.is_absolute() or ".." in rel.parts or "\n" in name or "\t" in name:
        raise FetchError(f"refusing unsafe archive member name: {name!r}")
    return rel


def _extracted_stat(dest: Path, zi: zipfile.ZipInfo):
    """stat of `dest` if it looks like the finished member, else None."""
    try:
        st = os.stat(dest)
    except OSError:
        return None
    return st if stat.S_ISREG(st.st_mode) and st.st_size == zi.file_size else None


def _extract_member(fd: int, zi: zipfile.ZipInfo, dest: Path, fallback) -> None:
    """Extract one member with positional reads, so threads can share `fd`."""
    if zi.flag_bits & 0x1:
        raise FetchError(f"{zi.filename}: encrypted members are not supported")
    tmp = dest.with_name(dest.name + TMP_SUFFIX)
    if zi.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        fallback(zi, tmp)
        os.replace(tmp, dest)
        return

    header = os.pread(fd, _LOCAL_HEADER.size, zi.header_offset)
    if len(header) != _LOCAL_HEADER.size or header[:4] != b"PK\x03\x04":
        raise zipfile.BadZipFile(f"{zi.filename}: no local header at offset {zi.header_offset}")
    fields = _LOCAL_HEADER.unpack(header)
    pos = zi.header_offset + _LOCAL_HEADER.size + fields[9] + fields[10]

    inflater = zlib.decompressobj(-15) if zi.compress_type == zipfile.ZIP_DEFLATED else None
    remaining, crc, written = zi.compress_size, 0, 0
    with open(tmp, "wb") as out:
        while remaining:
            chunk = os.pread(fd, min(remaining, 1 << 20), pos)
            if not chunk:
                raise zipfile.BadZipFile(f"{zi.filename}: archive truncated")
            pos += len(chunk)
            remaining -= len(chunk)
            data = inflater.decompress(chunk) if inflater else chunk
            crc = zlib.crc32(data, crc)
            written += len(data)
            out.write(data)
        if inflater:
            data = inflater.flush()
            crc = zlib.crc32(data, crc)
            written += len(data)
            out.write(data)
    if written != zi.file_size or crc != zi.CRC:
        os.unlink(tmp)
        raise zipfile.BadZipFile(f"{zi.filename}: size/CRC mismatch - the archive is corrupt")
    os.replace(tmp, dest)


def _write_manifest(path: Path, members) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for zi in members:
            if not zi.is_dir():
                fh.write(f"{zi.CRC:08x}\t{zi.file_size}\t{zi.filename}\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def extract(paths: Paths, workers: int, reclaim: str, keep_zip: bool, punch_step: int,
            min_free: int) -> None:
    if not paths.zip.exists():
        raise FetchError(f"{paths.zip} not found - run the download step first")

    with zipfile.ZipFile(paths.zip) as zf:
        members = sorted(zf.infolist(), key=lambda zi: zi.header_offset)
        central_dir_offset = zf.start_dir
        for zi in members:
            _safe_relpath(zi.filename)
        # Zip semantics: the last entry of a name wins. Extracting both in
        # parallel would also make them fight over one temporary file.
        last_index = {zi.filename: i for i, zi in enumerate(members)}
        superseded = {i for i, zi in enumerate(members) if last_index[zi.filename] != i}
        if superseded:
            log.warning("%s duplicate archive entries will be skipped", len(superseded))
        files = [zi for i, zi in enumerate(members) if not zi.is_dir() and i not in superseded]
        total_bytes = sum(zi.file_size for zi in files)
        log.info("archive holds %s files, %s uncompressed", len(files), _fmt_size(total_bytes))
        _write_manifest(paths.manifest, files)

        state = _read_json(paths.reclaim_state) or {}
        zip_stat = paths.zip.stat()
        already_punched = state.get("punched_upto", 0) > 0 and \
            (state.get("ino"), state.get("size")) == (zip_stat.st_ino, zip_stat.st_size)
        free = _free_bytes(paths.data)
        fits = free >= total_bytes + min_free
        use_reclaim = already_punched or reclaim == "always" or (reclaim == "auto" and not fits)
        if use_reclaim and keep_zip:
            raise FetchError("--keep-zip is impossible here: there is not enough free space to hold "
                             "the archive and the extracted data at the same time")
        if not use_reclaim and not fits:
            raise FetchError(f"not enough space to extract without reclaiming: {_fmt_size(free)} free, "
                             f"{_fmt_size(total_bytes + min_free)} needed. Use --reclaim auto")
        log.info("extract mode: %s (%s free)", "reclaim archive space while extracting"
                 if use_reclaim else "plain", _fmt_size(free))
        if use_reclaim:
            _require_hole_punching(paths.work)
            # Once holes exist there is no way back, so make sure the end state
            # fits before the first one. Members behind the hole are on disk
            # already; the archive's still-allocated blocks will be freed.
            upto = state["punched_upto"] if already_punched else 0
            remaining = sum(zi.file_size for zi in files if zi.header_offset >= upto)
            reclaimable = zip_stat.st_blocks * 512
            if free + reclaimable < remaining + min_free + punch_step:
                raise FetchError(
                    f"the extracted dataset will not fit: {_fmt_size(remaining)} still to extract, "
                    f"{_fmt_size(free)} free + {_fmt_size(reclaimable)} reclaimable from the archive")

        fd = os.open(paths.zip, os.O_RDWR if use_reclaim else os.O_RDONLY)
        try:
            reclaimer = (_Reclaimer(fd, paths.reclaim_state, punch_step, central_dir_offset)
                         if use_reclaim else None)
            punched_upto = reclaimer.done if reclaimer else 0
            synced_before_ns = reclaimer.synced_before_ns if reclaimer else 0

            def trusted(zi, st):
                # In reclaim mode a wrongly trusted file is lost for good once
                # its range is punched. Only files that a sync() is known to
                # have covered are trusted; anything newer is cheap to redo
                # because its archive range still exists.
                if not reclaimer or zi.header_offset < punched_upto:
                    return True
                return st.st_mtime_ns < synced_before_ns
            made_dirs = set()
            zf_lock = threading.Lock()

            def fallback(zi, tmp):
                with zf_lock, zf.open(zi) as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)

            def job(index, zi):
                dest = paths.data / zi.filename
                if index in superseded:
                    return 0
                if zi.is_dir():
                    os.makedirs(dest, exist_ok=True)
                    return 0
                st = _extracted_stat(dest, zi)
                if st and trusted(zi, st):
                    return 0
                if zi.header_offset < punched_upto:
                    raise FetchError(
                        f"{zi.filename} is missing but its archive range was already reclaimed; "
                        f"delete {paths.work} and download again")
                parent = dest.parent
                if parent not in made_dirs:
                    os.makedirs(parent, exist_ok=True)
                    made_dirs.add(parent)
                _extract_member(fd, zi, dest, fallback)
                return zi.file_size

            progress = _Progress("extract", total_bytes)
            done_bytes = 0
            pending = collections.deque()
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
            try:
                def drain_one():
                    nonlocal done_bytes
                    index, zi, fut = pending.popleft()
                    try:
                        fut.result()
                    except (OSError, zipfile.BadZipFile, zlib.error) as exc:
                        raise FetchError(f"extracting {zi.filename} failed: {exc}") from exc
                    done_bytes += zi.file_size
                    progress.update(done_bytes, f" [{index + 1}/{len(members)} members]")
                    if index % 5000 == 0 and _free_bytes(paths.data) < min_free:
                        raise FetchError(f"free space fell below the {_fmt_size(min_free)} reserve; "
                                         "stopping (re-run to resume)")
                    if reclaimer:
                        # Results are drained in archive order, so everything
                        # before the next member's header is safely on disk.
                        nxt = members[index + 1].header_offset if index + 1 < len(members) \
                            else central_dir_offset
                        reclaimer.advance(nxt)

                window = workers * 64
                for index, zi in enumerate(members):
                    while len(pending) >= window:
                        drain_one()
                    pending.append((index, zi, pool.submit(job, index, zi)))
                while pending:
                    drain_one()
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
            progress.update(done_bytes, force=True)
        finally:
            os.close(fd)

    os.sync()
    _write_json_atomic(paths.complete, {"files": len(files), "bytes": total_bytes,
                                        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if keep_zip:
        log.info("extraction complete; archive kept at %s", paths.zip)
    else:
        paths.zip.unlink()
        paths.reclaim_state.unlink(missing_ok=True)
        log.info("extraction complete; archive removed")


# ----------------------------------------------------------------------- verify


def _file_crc(path: Path) -> int:
    crc = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                return crc
            crc = zlib.crc32(chunk, crc)


def verify(paths: Paths, check_crc: bool, workers: int) -> None:
    if not paths.manifest.exists():
        raise FetchError(f"{paths.manifest} not found - nothing has been extracted yet")

    def check(line: str):
        crc_hex, size, name = line.rstrip("\n").split("\t", 2)
        path = paths.data / name
        try:
            st = os.stat(path)
        except OSError:
            return name, "missing"
        if st.st_size != int(size):
            return name, f"size {st.st_size} != {size}"
        if check_crc and _file_crc(path) != int(crc_hex, 16):
            return name, "crc mismatch"
        return name, None

    bad = total = 0
    with open(paths.manifest, encoding="utf-8") as fh, \
            concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        # Executor.map queues its whole input up front; batching keeps ~1M
        # manifest lines from turning into ~1M live futures.
        while True:
            batch = list(itertools.islice(fh, 10000))
            if not batch:
                break
            for name, problem in pool.map(check, batch):
                total += 1
                if problem:
                    bad += 1
                    if bad <= 20:
                        log.error("%s: %s", name, problem)
    if bad:
        raise FetchError(f"verification FAILED: {bad} of {total} files are wrong (first 20 listed)")
    log.info("verification OK: %s files%s", total, " (sizes + CRC32)" if check_crc else " (sizes)")


# -------------------------------------------------------------------------- cli


def preflight(paths: Paths, auth: Auth, competition: str, min_free: int) -> None:
    log.info("credentials: %s", auth.source)
    if paths.complete.exists():
        log.info("dataset already fully extracted: %s", _read_json(paths.complete))
        return
    url = resolve_archive_url(auth, competition)
    log.info("competition access OK (rules accepted)")
    total, _etag, ranges = probe_archive(url)
    free = _free_bytes(paths.work)
    log.info("archive size: %s | resumable download: %s | free on %s: %s",
             _fmt_size(total) if total else "unknown", "yes" if ranges else "NO",
             paths.data, _fmt_size(free))
    have = paths.part.stat().st_size if paths.part.exists() else 0
    if total and not paths.zip.exists() and free < total - have + min_free:
        raise FetchError(f"the archive does not fit: {_fmt_size(free)} free, "
                         f"{_fmt_size(total - have + min_free)} needed")
    log.info("preflight OK")


def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler(sys.stderr),
                logging.FileHandler(log_dir / time.strftime("fetch-%Y%m%d-%H%M%S.log"))]
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    for handler in handlers:
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    default_config = os.environ.get("KAGGLE_CONFIG_DIR") or str(script_dir / ".kaggle")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["preflight", "download", "extract", "verify", "all"])
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--config-dir", default=default_config,
                        help="directory holding access_token or kaggle.json (default: %(default)s)")
    parser.add_argument("--min-free-gib", type=float, default=20.0,
                        help="stop before free space drops below this (shared disk courtesy)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reclaim", choices=["auto", "always", "never"], default="auto",
                        help="free archive space while extracting (the archive is consumed)")
    parser.add_argument("--punch-step-gib", type=float, default=8.0)
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument("--crc", action="store_true", help="verify: also re-read and CRC every file")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    paths = Paths(Path(args.data_dir), args.competition)
    paths.work.mkdir(parents=True, exist_ok=True)
    _setup_logging(paths.logs)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    min_free = int(args.min_free_gib * GIB)

    lock_fh = open(paths.lock, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock_fh.close()
        if exc.errno in (errno.EAGAIN, errno.EACCES):
            log.error("another kaggle_fetch run is already working on %s", paths.data)
            return 3
        raise

    try:
        cmd = args.command
        if paths.complete.exists() and paths.zip.exists() and not args.keep_zip \
                and cmd in ("extract", "all"):
            log.info("removing the archive left behind by an interrupted finish")
            paths.zip.unlink()
            paths.reclaim_state.unlink(missing_ok=True)
        if cmd == "extract" and paths.complete.exists() and not paths.zip.exists():
            log.info("already extracted: %s", _read_json(paths.complete))
            return 0
        needs_network = cmd in ("preflight", "download") or (
            cmd == "all" and not paths.zip.exists() and not paths.complete.exists())
        auth = load_auth(Path(args.config_dir)) if needs_network else None

        if cmd in ("preflight", "all") and needs_network:
            preflight(paths, auth, args.competition, min_free)
        if cmd in ("download", "all") and needs_network:
            if paths.complete.exists():
                log.info("already complete; delete %s to fetch again", paths.complete)
            else:
                download(paths, lambda: resolve_archive_url(auth, args.competition), min_free)
        if cmd == "extract" or (cmd == "all" and not paths.complete.exists()):
            extract(paths, args.workers, args.reclaim, args.keep_zip,
                    int(args.punch_step_gib * GIB), min_free)
        if cmd in ("verify", "all"):
            verify(paths, args.crc, args.workers)
        return 0
    except FetchError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("interrupted - re-run the same command to resume")
        return 130
    except (urllib.error.URLError, http.client.HTTPException) as exc:
        log.error("network problem talking to Kaggle: %r - re-run to retry", exc)
        return 4
    finally:
        lock_fh.close()


if __name__ == "__main__":
    sys.exit(main())
