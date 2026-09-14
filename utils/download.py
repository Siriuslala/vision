"""Download a dataset snapshot using HTTP by default to avoid Xet CAS failures."""

import argparse
import os
from pathlib import Path
import sys
import time


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Dataset ID, e.g. allenai/pixmo-count")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-workers", type=positive_int, default=2)
    parser.add_argument("--download-timeout", type=positive_int, default=120)
    parser.add_argument("--etag-timeout", type=positive_int, default=60)
    parser.add_argument("--attempts", type=positive_int, default=3)
    parser.add_argument("--use-xet", action="store_true", help="Opt into the Xet transfer backend")
    args = parser.parse_args()
    if not args.local_dir.is_absolute():
        parser.error("--local-dir must be an absolute path")

    # Hub constants are read at import time; configure before importing the library.
    os.environ["HF_HUB_DISABLE_XET"] = "0" if args.use_xet else "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.download_timeout)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(args.etag_timeout)

    import httpx
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    print(f"Downloading dataset {args.repo_id}@{args.revision} to {args.local_dir} "
          f"using {'Xet' if args.use_xet else 'HTTP'}", flush=True)
    for attempt in range(1, args.attempts + 1):
        try:
            result = snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                local_dir=args.local_dir,
                max_workers=args.max_workers,
                etag_timeout=args.etag_timeout,
            )
        except (httpx.TransportError, HfHubHTTPError) as error:
            # Authentication, missing repositories and local I/O errors fail immediately.
            if isinstance(error, HfHubHTTPError):
                status = error.response.status_code if error.response is not None else None
                if status not in (408, 429, 500, 502, 503, 504):
                    raise
            if attempt == args.attempts:
                raise
            delay = min(5 * 2 ** (attempt - 1), 60)
            print(f"Temporary download failure ({type(error).__name__}); "
                  f"retrying in {delay}s (attempt {attempt + 1}/{args.attempts}).",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
        else:
            print(f"Dataset snapshot available at: {result}", flush=True)
            return


if __name__ == "__main__":
    main()
