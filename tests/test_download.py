import os
import sys
import unittest
from unittest.mock import patch

import httpx
from huggingface_hub.errors import HfHubHTTPError

from utils.download import main


class DownloadTests(unittest.TestCase):
    def run_download(self, effects):
        argv = ["download", "--repo-id", "allenai/pixmo-count",
                "--local-dir", "/tmp/download-test", "--attempts", "3"]
        with patch.object(sys, "argv", argv), patch.dict(os.environ), \
             patch("huggingface_hub.snapshot_download", side_effect=effects) as download, \
             patch("utils.download.time.sleep"), patch("builtins.print"):
            main()
        return download

    def test_transport_failure_recovers(self):
        download = self.run_download([httpx.ReadError("truncated response"), "/tmp/download-test"])
        self.assertEqual(download.call_count, 2)

    def test_retry_exhaustion_preserves_failure(self):
        error = httpx.ReadTimeout("timeout")
        with self.assertRaises(httpx.ReadTimeout) as raised:
            self.run_download([error] * 3)
        self.assertIs(raised.exception, error)

    def test_authentication_failure_is_not_retried(self):
        response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co"))
        with self.assertRaises(HfHubHTTPError):
            self.run_download([HfHubHTTPError("forbidden", response=response), "/tmp/unexpected"])

    def test_local_io_failure_is_not_retried(self):
        with self.assertRaises(PermissionError):
            self.run_download([PermissionError("read-only destination"), "/tmp/unexpected"])
