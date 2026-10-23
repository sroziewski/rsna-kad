"""Tests for scripts/kaggle_fetch.py. Linux only (flock, fallocate).

Run on the server:  TMPDIR=<dir on the target filesystem> python -m pytest tests -q
"""
import errno
import fcntl
import http.server
import io
import json
import os
import random
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_fetch as kf  # noqa: E402

COMP = "demo-comp"


def _random_bytes(n, seed):
    return random.Random(seed).randbytes(n)


def build_zip(path, big=False):
    """Mixed stored/deflated members in nested dirs; returns {name: bytes}."""
    size = 256 * 1024 if big else 3000
    content = {"train.csv": b"id,label\n1,0\n", "empty.bin": b""}
    for i in range(40 if big else 6):
        content[f"train/study{i % 3}/series{i}/img{i}.dcm"] = _random_bytes(size, i)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(zipfile.ZipInfo("train/"), b"")
        for i, (name, data) in enumerate(content.items()):
            method = zipfile.ZIP_DEFLATED if i % 2 else zipfile.ZIP_STORED
            zf.writestr(name, data, compress_type=method)
    return content


class FakeKaggle:
    """Kaggle API (redirect issuer) + signed-URL blob store on one local port."""

    def __init__(self, blob):
        self.blob = blob
        self.etag = '"v1"'
        self.api_status = {"rpc": 302, "rest": 302}
        self.valid_sig = 1
        self.resolves = 0
        self.drop_after = []          # per blob request: bytes to send before cutting the socket
        self.ignore_range = False
        self.unstable_etag = False
        self.blob_requests = []       # (range header, authorization header)
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _api(self, kind):
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                status = outer.api_status[kind]
                if self.headers.get("Authorization") != "Bearer good-token":
                    status = 401
                body = json.dumps({"error": {"code": status, "message": f"msg-{status}"}}).encode()
                self.send_response(status)
                if status == 302:
                    outer.resolves += 1
                    self.send_header("Location", f"/blob?sig={outer.valid_sig}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self._api("rpc")

            def do_GET(self):
                if not self.path.startswith("/blob"):
                    return self._api("rest")
                rng = self.headers.get("Range")
                outer.blob_requests.append((rng, self.headers.get("Authorization")))
                if self.path != f"/blob?sig={outer.valid_sig}":
                    self.send_response(403)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                blob = outer.blob
                start = 0
                if rng and not outer.ignore_range:
                    first, _, last = rng[len("bytes="):].partition("-")
                    start = int(first)
                    end = int(last) if last else len(blob) - 1
                    if start >= len(blob):
                        self.send_response(416)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    payload = blob[start:end + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(blob)}")
                else:
                    payload = blob
                    self.send_response(200)
                etag = f'"req-{len(outer.blob_requests)}"' if outer.unstable_etag else outer.etag
                self.send_header("ETag", etag)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if outer.drop_after:
                    cut = outer.drop_after.pop(0)
                    self.wfile.write(payload[:cut])
                    self.wfile.flush()
                    self.close_connection = True
                    self.connection.close()
                    return
                self.wfile.write(payload)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.paths = kf.Paths(self.tmp / "data", COMP)
        self.paths.work.mkdir(parents=True)
        patcher = mock.patch.object(kf.time, "sleep", lambda s: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def start_server(self, blob):
        srv = FakeKaggle(blob)
        self.addCleanup(srv.close)
        for name, value in (("API_RPC_BASE", srv.base + "/rpc"), ("API_REST_BASE", srv.base + "/rest")):
            patcher = mock.patch.object(kf, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return srv

    def assert_tree(self, content):
        for name, data in content.items():
            self.assertEqual((self.paths.data / name).read_bytes(), data, name)
        leftovers = [p for p in self.paths.data.rglob("*" + kf.TMP_SUFFIX)]
        self.assertEqual(leftovers, [])


GOOD = kf.Auth("Bearer good-token", "test")


class AuthTests(Base):
    def test_env_token_wins(self):
        (self.tmp / "access_token").write_text("file-token")
        auth = kf.load_auth(self.tmp, {"KAGGLE_API_TOKEN": " env-token \n"})
        self.assertEqual(auth.header, "Bearer env-token")

    def test_token_file_is_stripped(self):
        token = self.tmp / "access_token"
        token.write_text("KGAT_abc\r\n")
        token.chmod(0o600)
        self.assertEqual(kf.load_auth(self.tmp, {}).header, "Bearer KGAT_abc")

    def test_exposed_token_file_warns(self):
        token = self.tmp / "access_token"
        token.write_text("t")
        token.chmod(0o644)
        with self.assertLogs(kf.log, "WARNING"):
            kf.load_auth(self.tmp, {})

    def test_legacy_json_is_basic_auth(self):
        legacy = self.tmp / "kaggle.json"
        legacy.write_text(json.dumps({"username": "u", "key": "k"}))
        legacy.chmod(0o600)
        self.assertEqual(kf.load_auth(self.tmp, {}).header, "Basic dTpr")  # base64("u:k")

    def test_no_credentials(self):
        with self.assertRaisesRegex(kf.FetchError, "no Kaggle credentials"):
            kf.load_auth(self.tmp, {})


class ResolveTests(Base):
    def test_redirect_is_captured_not_followed(self):
        srv = self.start_server(b"x" * 10)
        url = kf.resolve_archive_url(GOOD, COMP)
        self.assertEqual(url, srv.base + "/blob?sig=1")
        self.assertEqual(srv.blob_requests, [])

    def test_bad_token(self):
        self.start_server(b"")
        with self.assertRaisesRegex(kf.FetchError, "rejected the credentials"):
            kf.resolve_archive_url(kf.Auth("Bearer nope", "test"), COMP)

    def test_rules_not_accepted(self):
        srv = self.start_server(b"")
        srv.api_status["rpc"] = 403
        with self.assertRaisesRegex(kf.FetchError, f"competitions/{COMP}/rules"):
            kf.resolve_archive_url(GOOD, COMP)

    def test_falls_back_to_rest_route(self):
        srv = self.start_server(b"")
        srv.api_status["rpc"] = 404
        self.assertTrue(kf.resolve_archive_url(GOOD, COMP).endswith("/blob?sig=1"))

    def test_all_routes_fail(self):
        srv = self.start_server(b"")
        srv.api_status = {"rpc": 404, "rest": 404}
        with self.assertRaisesRegex(kf.FetchError, "could not obtain"):
            kf.resolve_archive_url(GOOD, COMP)


class DownloadTests(Base):
    def setUp(self):
        super().setUp()
        src = self.tmp / "src.zip"
        build_zip(src)
        self.blob = src.read_bytes()
        self.srv = self.start_server(self.blob)
        patcher = mock.patch.object(kf, "CHUNK", 1024)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_download(self, min_free=0, **kw):
        kf.download(self.paths, lambda: kf.resolve_archive_url(GOOD, COMP), min_free, **kw)

    def test_full_download_never_leaks_credentials_to_storage(self):
        self.run_download()
        self.assertEqual(self.paths.zip.read_bytes(), self.blob)
        self.assertFalse(self.paths.part.exists())
        self.assertFalse(self.paths.part_meta.exists())
        self.assertTrue(all(auth is None for _, auth in self.srv.blob_requests))

    def test_resumes_partial_file(self):
        self.paths.part.write_bytes(self.blob[:5000])
        kf._write_json_atomic(self.paths.part_meta, {"etag": '"v1"', "total": len(self.blob)})
        self.run_download()
        self.assertEqual(self.paths.zip.read_bytes(), self.blob)
        self.assertEqual(self.srv.blob_requests[0][0], "bytes=5000-")

    def test_survives_dropped_connections(self):
        self.srv.drop_after = [3000, 0, 2500]
        self.run_download()
        self.assertEqual(self.paths.zip.read_bytes(), self.blob)
        self.assertGreaterEqual(len(self.srv.blob_requests), 4)

    def test_expired_url_is_renewed(self):
        self.srv.drop_after = [4000]
        real_resolve = kf.resolve_archive_url
        calls = []

        def resolve():
            url = real_resolve(GOOD, COMP)
            calls.append(url)
            if len(calls) == 1:
                self.srv.valid_sig = 2   # the URL we just handed out dies after first use
            return url

        kf.download(self.paths, resolve, 0)
        self.assertEqual(self.paths.zip.read_bytes(), self.blob)
        self.assertEqual(len(calls), 2)

    def test_changed_remote_archive_restarts(self):
        self.paths.part.write_bytes(b"STALE" * 1000)
        kf._write_json_atomic(self.paths.part_meta, {"etag": '"old"', "total": len(self.blob)})
        self.run_download()
        self.assertEqual(self.paths.zip.read_bytes(), self.blob)

    def test_server_without_range_support_restarts(self):
        self.srv.ignore_range = True
        self.paths.part.write_bytes(self.blob[:5000])
        kf._write_json_atomic(self.paths.part_meta, {"etag": '"v1"', "total": len(self.blob)})
        self.run_download()
        self.assertEqual(self.paths.zip.read_bytes(), self.blob)

    def test_partial_without_metadata_restarts(self):
        self.paths.part.write_bytes(b"garbage")
        self.run_download()
        self.assertEqual(self.paths.zip.read_bytes(), self.blob)

    def test_refuses_when_disk_too_small(self):
        with self.assertRaisesRegex(kf.FetchError, "not enough disk space"):
            self.run_download(min_free=1 << 60)
        self.assertFalse(self.paths.zip.exists())

    def test_gives_up_without_progress(self):
        self.srv.drop_after = [0] * 50
        with self.assertRaisesRegex(kf.FetchError, "giving up"):
            self.run_download(max_failures=3)

    def test_unstable_remote_identity_is_bounded(self):
        self.srv.unstable_etag = True
        self.srv.drop_after = [3000] * 20
        with self.assertRaisesRegex(kf.FetchError, "cannot be resumed"):
            self.run_download()

    def test_new_archive_discards_old_reclaim_state(self):
        kf._write_json_atomic(self.paths.reclaim_state, {"punched_upto": 123, "ino": 1, "size": 1})
        self.run_download()
        self.assertFalse(self.paths.reclaim_state.exists())

    def test_content_range_with_unknown_total(self):
        self.assertEqual(kf._parse_content_range("bytes 0-0/*"), (0, None))
        self.assertEqual(kf._parse_content_range("bytes 5-9/10"), (5, 10))
        self.assertIsNone(kf._parse_content_range("items 0-1/2"))

    def test_non_zip_payload_is_rejected(self):
        self.srv.blob = b"<html>login</html>" * 100
        with self.assertRaisesRegex(kf.FetchError, "not a zip"):
            self.run_download()


class ExtractTests(Base):
    def extract(self, reclaim="never", keep_zip=False, step=1 << 20, min_free=0, workers=4):
        kf.extract(self.paths, workers, reclaim, keep_zip, step, min_free)

    def test_plain_extract(self):
        content = build_zip(self.paths.zip)
        self.extract()
        self.assert_tree(content)
        self.assertTrue((self.paths.data / "train").is_dir())
        self.assertFalse(self.paths.zip.exists())
        self.assertEqual(kf._read_json(self.paths.complete)["files"], len(content))

    def test_keep_zip(self):
        build_zip(self.paths.zip)
        self.extract(keep_zip=True)
        self.assertTrue(self.paths.zip.exists())

    def test_resume_skips_good_and_redoes_wrong_size(self):
        content = build_zip(self.paths.zip)
        self.extract(keep_zip=True)
        good = self.paths.data / "train.csv"
        before = good.stat().st_mtime_ns
        broken = self.paths.data / "train/study0/series0/img0.dcm"
        broken.write_bytes(b"short")
        os.remove(self.paths.data / "train/study1/series1/img1.dcm")
        self.extract(keep_zip=True)
        self.assert_tree(content)
        self.assertEqual(good.stat().st_mtime_ns, before)

    def test_bzip2_member_uses_fallback(self):
        with zipfile.ZipFile(self.paths.zip, "w") as zf:
            zf.writestr("a.txt", b"hello" * 1000, compress_type=zipfile.ZIP_BZIP2)
        self.extract()
        self.assertEqual((self.paths.data / "a.txt").read_bytes(), b"hello" * 1000)

    def test_path_traversal_is_refused(self):
        with zipfile.ZipFile(self.paths.zip, "w") as zf:
            zf.writestr("ok.txt", b"1")
            zf.writestr("../evil.txt", b"2")
        with self.assertRaisesRegex(kf.FetchError, "unsafe"):
            self.extract()
        self.assertFalse((self.tmp / "evil.txt").exists())
        self.assertFalse((self.paths.data / "ok.txt").exists())

    def test_corrupt_member_is_detected(self):
        with zipfile.ZipFile(self.paths.zip, "w") as zf:
            zf.writestr("a.bin", b"A" * 5000, compress_type=zipfile.ZIP_STORED)
        raw = bytearray(self.paths.zip.read_bytes())
        raw[raw.index(b"AAAA") + 100] ^= 0xFF
        self.paths.zip.write_bytes(raw)
        with self.assertRaisesRegex(kf.FetchError, "corrupt"):
            self.extract()
        self.assertFalse((self.paths.data / "a.bin").exists())
        self.assertTrue(self.paths.zip.exists())

    def test_streamed_zip64_members_with_data_descriptors(self):
        class Unseekable(io.RawIOBase):
            def __init__(self, fh):
                self.fh = fh

            def writable(self):
                return True

            def write(self, data):
                return self.fh.write(data)

        content = {f"d/f{i}.bin": _random_bytes(5000, i) for i in range(4)}
        with open(self.paths.zip, "wb") as fh, zipfile.ZipFile(Unseekable(fh), "w") as zf:
            for i, (name, data) in enumerate(content.items()):
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED if i % 2 else zipfile.ZIP_STORED
                with zf.open(info, "w", force_zip64=True) as dst:
                    dst.write(data)
        with zipfile.ZipFile(self.paths.zip) as zf:
            self.assertTrue(all(zi.flag_bits & 0x08 for zi in zf.infolist()))
        self.extract()
        self.assert_tree(content)

    def test_duplicate_names_last_entry_wins(self):
        with self.assertWarns(UserWarning), zipfile.ZipFile(self.paths.zip, "w") as zf:
            zf.writestr("a.txt", b"first")
            zf.writestr("a.txt", b"second!")
        self.extract()
        self.assertEqual((self.paths.data / "a.txt").read_bytes(), b"second!")
        kf.verify(self.paths, True, 2)

    def test_unsupported_hole_punching_fails_before_any_damage(self):
        content = build_zip(self.paths.zip)
        unsupported = OSError(errno.EOPNOTSUPP, "Operation not supported")
        with mock.patch.object(kf, "punch_hole", side_effect=unsupported):
            with self.assertRaisesRegex(kf.FetchError, "cannot give archive space back"):
                self.extract(reclaim="always")
        self.assertFalse(self.paths.reclaim_state.exists())
        self.extract(reclaim="never")
        self.assert_tree(content)

    def test_never_mode_refuses_when_space_is_short(self):
        build_zip(self.paths.zip)
        with self.assertRaisesRegex(kf.FetchError, "not enough space"):
            self.extract(reclaim="never", min_free=1 << 60)

    def test_keep_zip_conflicts_with_reclaim(self):
        build_zip(self.paths.zip)
        with self.assertRaisesRegex(kf.FetchError, "--keep-zip is impossible"):
            self.extract(reclaim="always", keep_zip=True)


class ReclaimTests(Base):
    def setUp(self):
        super().setUp()
        self.content = build_zip(self.paths.zip, big=True)
        try:
            probe = self.tmp / "probe"
            probe.write_bytes(b"\0" * 8192)
            fd = os.open(probe, os.O_RDWR)
            try:
                kf.punch_hole(fd, 0, 4096)
            finally:
                os.close(fd)
        except OSError as exc:
            if exc.errno in (errno.EOPNOTSUPP, errno.ENOSYS):
                self.skipTest("filesystem cannot punch holes")
            raise

    def extract(self, **kw):
        kf.extract(self.paths, kw.pop("workers", 4), kw.pop("reclaim", "always"), False, 1 << 20, 0)

    def test_reclaim_extracts_everything(self):
        self.extract()
        self.assert_tree(self.content)
        self.assertFalse(self.paths.zip.exists())
        self.assertFalse(self.paths.reclaim_state.exists())

    def test_auto_picks_reclaim_when_space_is_short(self):
        with mock.patch.object(kf, "_free_bytes", return_value=2 << 20),                 self.assertLogs(kf.log) as logs:
            kf.extract(self.paths, 4, "auto", False, 1 << 20, 0)
        self.assertTrue(any("reclaimed archive space" in line for line in logs.output))
        self.assert_tree(self.content)

    def test_refuses_up_front_when_the_result_cannot_fit(self):
        blocks = self.paths.zip.stat().st_blocks
        with mock.patch.object(kf, "_free_bytes", return_value=0):
            with self.assertRaisesRegex(kf.FetchError, "will not fit"):
                self.extract()
        self.assertEqual(self.paths.zip.stat().st_blocks, blocks)
        self.assertFalse(self.paths.reclaim_state.exists())

    def test_failed_punch_does_not_claim_the_range_was_reclaimed(self):
        real = kf.punch_hole
        calls = []

        def flaky(fd, offset, length):
            calls.append(offset)
            if len(calls) > 1:              # the first call is the capability probe
                raise OSError(errno.EIO, "boom")
            real(fd, offset, length)

        with mock.patch.object(kf, "punch_hole", flaky):
            with self.assertRaisesRegex(kf.FetchError, "could not reclaim"):
                self.extract()
        self.assertEqual(kf._read_json(self.paths.reclaim_state)["punched_upto"], 0)
        self.extract()
        self.assert_tree(self.content)

    def _garbage_tree(self):
        for name, data in self.content.items():
            path = self.paths.data / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"G" * len(data))

    def test_files_not_covered_by_a_sync_are_redone(self):
        self._garbage_tree()
        self.extract()
        self.assert_tree(self.content)

    def test_files_older_than_the_last_sync_are_trusted(self):
        self._garbage_tree()
        hour_ago = kf.time.time() - 3600
        victim = self.paths.data / "train/study0/series0/img0.dcm"
        os.utime(victim, (hour_ago, hour_ago))
        self.extract()
        self.assertEqual(victim.read_bytes(), b"G" * len(self.content["train/study0/series0/img0.dcm"]))
        with self.assertRaisesRegex(kf.FetchError, "FAILED"):
            kf.verify(self.paths, True, 2)

    def test_state_of_another_archive_is_ignored(self):
        kf._write_json_atomic(self.paths.reclaim_state, {"punched_upto": 1 << 40, "ino": 1, "size": 1})
        self.extract()
        self.assert_tree(self.content)

    def _fail_midway(self):
        """Block one late member so the run dies after holes were punched."""
        names = [n for n in self.content if n.endswith(".dcm")]
        blocker = self.paths.data / names[-3]
        blocker.mkdir(parents=True)
        blocks_before = self.paths.zip.stat().st_blocks
        with self.assertRaises(kf.FetchError):
            self.extract(workers=1)
        state = kf._read_json(self.paths.reclaim_state)
        self.assertGreater(state["punched_upto"], 0)
        self.assertLess(self.paths.zip.stat().st_blocks, blocks_before)
        self.assertFalse(self.paths.complete.exists())
        blocker.rmdir()
        return names

    def test_resume_after_failure_in_reclaim_mode(self):
        self._fail_midway()
        # Even "never" must stay in reclaim mode: the archive already has holes.
        self.extract(reclaim="never")
        self.assert_tree(self.content)
        self.assertFalse(self.paths.zip.exists())

    def test_lost_file_behind_the_hole_is_reported(self):
        names = self._fail_midway()
        os.remove(self.paths.data / names[0])
        with self.assertRaisesRegex(kf.FetchError, "already reclaimed"):
            self.extract()


class VerifyTests(Base):
    def setUp(self):
        super().setUp()
        self.content = build_zip(self.paths.zip)
        kf.extract(self.paths, 2, "never", False, 1 << 20, 0)

    def test_ok(self):
        kf.verify(self.paths, True, 2)

    def test_missing_file(self):
        os.remove(self.paths.data / "train.csv")
        with self.assertRaisesRegex(kf.FetchError, "1 of"):
            kf.verify(self.paths, False, 2)

    def test_silent_corruption_needs_crc(self):
        target = self.paths.data / "train/study0/series0/img0.dcm"
        data = bytearray(target.read_bytes())
        data[10] ^= 0xFF
        target.write_bytes(data)
        kf.verify(self.paths, False, 2)
        with self.assertRaisesRegex(kf.FetchError, "FAILED"):
            kf.verify(self.paths, True, 2)


class MainTests(Base):
    def setUp(self):
        super().setUp()
        src = self.tmp / "src.zip"
        self.content = build_zip(src)
        self.srv = self.start_server(src.read_bytes())
        self.config = self.tmp / "cfg"
        self.config.mkdir()
        token = self.config / "access_token"
        token.write_text("good-token\n")
        token.chmod(0o600)
        self.argv = ["--data-dir", str(self.paths.data), "--config-dir", str(self.config),
                     "--competition", COMP, "--min-free-gib", "0"]

    def test_all_end_to_end_then_idempotent_offline(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KAGGLE_API_TOKEN", None)
            self.assertEqual(kf.main(["all"] + self.argv), 0)
            self.assert_tree(self.content)
            self.assertFalse(self.paths.zip.exists())
            resolves = self.srv.resolves
            self.assertEqual(kf.main(["all"] + self.argv), 0)
            self.assertEqual(self.srv.resolves, resolves)
        logged = "".join(p.read_text() for p in self.paths.logs.glob("*.log"))
        self.assertIn("download complete", logged)
        self.assertNotIn("good-token", logged)
        self.assertNotIn("sig=", logged)

    def test_archive_left_by_interrupted_finish_is_removed(self):
        self.assertEqual(kf.main(["all"] + self.argv), 0)
        build_zip(self.paths.zip)
        self.assertEqual(kf.main(["all"] + self.argv), 0)
        self.assertFalse(self.paths.zip.exists())

    def test_missing_credentials_exit_code(self):
        (self.config / "access_token").unlink()
        with mock.patch.dict(os.environ, {}, clear=False):
            for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
                os.environ.pop(var, None)
            self.assertEqual(kf.main(["preflight"] + self.argv), 2)

    def test_second_instance_is_refused(self):
        with open(self.paths.lock, "w") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            self.assertEqual(kf.main(["verify"] + self.argv), 3)


if __name__ == "__main__":
    unittest.main()
