"""Fetch the official MSR-VTT 9K training videos without changing the split."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import json
from pathlib import Path
import struct
import threading
import time
import zipfile
import zlib

import requests


SOURCE = "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip"
ROOT = Path(__file__).resolve().parent
SESSIONS = threading.local()


def get_range(begin, end):
    if not hasattr(SESSIONS, "session"):
        SESSIONS.session = requests.Session()
    response = SESSIONS.session.get(
        SOURCE, headers={"Range": f"bytes={begin}-{end}"}, timeout=(20, 90)
    )
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError("The archive server did not honor the byte range")
    if not response.headers.get("Content-Range", "").startswith(f"bytes {begin}-"):
        raise RuntimeError("The archive server returned the wrong byte range")
    return response.content


def download_video(item):
    target = ROOT / "data" / "MSRVTT" / "vids" / Path(item.filename).name
    if target.is_file() and target.stat().st_size == item.file_size:
        if zlib.crc32(target.read_bytes()) == item.CRC:
            return
    end = item.header_offset + item.compress_size + 65536
    for attempt in range(4):
        try:
            raw = get_range(item.header_offset, end)
            if raw[:4] != b"PK\x03\x04":
                raise RuntimeError(f"Invalid local ZIP header for {item.filename}")
            name_size, extra_size = struct.unpack_from("<HH", raw, 26)
            offset = 30 + name_size + extra_size
            compressed = raw[offset:offset + item.compress_size]
            if item.compress_type == zipfile.ZIP_DEFLATED:
                content = zlib.decompress(compressed, -15)
            elif item.compress_type == zipfile.ZIP_STORED:
                content = compressed
            else:
                raise RuntimeError(f"Unsupported ZIP compression {item.compress_type}")
            if len(content) != item.file_size or zlib.crc32(content) != item.CRC:
                raise RuntimeError(f"Video CRC mismatch for {item.filename}")
            temporary = target.with_suffix(".mp4.part")
            temporary.write_bytes(content)
            temporary.replace(target)
            return
        except (requests.RequestException, RuntimeError, zlib.error):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def main():
    start = time.monotonic()
    response = requests.get(
        SOURCE, headers={"Range": "bytes=-65536"}, timeout=(20, 90)
    )
    response.raise_for_status()
    tail = response.content
    position = tail.rfind(b"PK\x06\x06")
    if position < 0:
        raise RuntimeError("The official ZIP64 directory was not found")
    fields = struct.unpack("<4sQ2H2I4Q", tail[position:position + 56])
    count, size, offset = fields[7], fields[8], fields[9]
    directory = get_range(offset, offset + size - 1)
    end_record = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, count, count, size, 0, 0)
    archive = zipfile.ZipFile(io.BytesIO(directory + end_record))
    split = ROOT / "data" / "MSRVTT" / "MSRVTT_train.9k.csv"
    with split.open(encoding="utf-8") as handle:
        expected = {row["video_id"] + ".mp4" for row in csv.DictReader(handle)}
    selected = [item for item in archive.infolist() if Path(item.filename).name in expected]
    if len(selected) != len(expected) or len(expected) != 9000:
        raise RuntimeError("Archive contents do not match the official 9K training split")
    (ROOT / "data" / "MSRVTT" / "vids").mkdir(parents=True, exist_ok=True)
    print(f"Preparing {len(selected)} training videos", flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        tasks = [pool.submit(download_video, item) for item in selected]
        for completed, task in enumerate(as_completed(tasks), 1):
            task.result()
            if completed == 1 or completed % 200 == 0:
                print(f"TRAIN VIDEOS {completed}/9000 elapsed {time.monotonic()-start:.1f}s", flush=True)
    manifest = {
        "source": SOURCE,
        "split": str(split.relative_to(ROOT)),
        "count": len(selected),
        "crc32_verified": True,
        "files": [
            {"name": Path(item.filename).name, "bytes": item.file_size, "crc32": f"{item.CRC:08x}"}
            for item in selected
        ],
    }
    path = ROOT / "downloads" / "msrvtt_9k_train_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"COMPLETE {len(selected)} training videos in {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
