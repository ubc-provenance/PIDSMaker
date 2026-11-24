"""Downloads the network flows of DARPA OpTC from the official Google Drive of the dataset.

The official release (https://github.com/FiveDirections/OpTC-data) contains an `ecar-bro` folder
with the eCAR FLOW-START events of all the hosts of the exercise, annotated with the Bro (Zeek)
id of the flow. This script downloads its `benign` and `evaluation` folders (the `short` folder
contains events with missing data), and the red team ground truth report.

Usage:
    python dataset_preprocessing/optc/download_optc_net.py /data/optc_net

Files already downloaded are skipped, so the script can be run again to resume a download
interrupted by Google Drive quotas.
"""

import argparse
import importlib
import os
import time

import gdown
import requests

OPTC_ROOT_FOLDER = "1n3kkS3KR31KUegn42yk3-e6JkZvf0Caa"
ECAR_BRO_SUBFOLDERS = ["benign", "evaluation"]
GROUND_TRUTH_FILE = "OpTCRedTeamGroundTruth.pdf"
FOLDER_MIME = "application/vnd.google-apps.folder"

_parse_folder = importlib.import_module("gdown.download_folder")._parse_google_drive_file


def list_folder(session, folder_id, retries=5):
    """[(id, name, mime type)] of the items of a public Google Drive folder."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    for attempt in range(retries):
        try:
            response = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            response.raise_for_status()
            _, items = _parse_folder(url=url, content=response.text)
            return list(items)
        except Exception as e:  # network errors, throttling
            if attempt == retries - 1:
                raise
            print(f"Listing {folder_id} failed ({e}), retrying...")
            time.sleep(10 * (attempt + 1))


def walk(session, folder_id, path):
    """Yields (file id, relative path) of all files under a folder."""
    for item_id, name, mime in list_folder(session, folder_id):
        item_path = os.path.join(path, name)
        if mime == FOLDER_MIME:
            yield from walk(session, item_id, item_path)
        else:
            yield item_id, item_path


def _download_direct(session, file_id, path):
    """Streams a file from the direct download endpoint of Google Drive (handles large files
    without the confirmation page), resuming a partial download."""
    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "confirm": "t"}
    start = os.path.getsize(path) if os.path.exists(path) else 0
    headers = {"User-Agent": "Mozilla/5.0", **({"Range": f"bytes={start}-"} if start else {})}
    with session.get(url, params=params, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError("Google Drive returned a web page instead of the file (quota?)")
        mode = "ab" if start and response.status_code == 206 else "wb"
        with open(path, mode) as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def download(session, file_id, out_path, retries=5):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return "skipped"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".part"
    for attempt in range(retries):
        try:
            _download_direct(session, file_id, tmp_path)
            os.replace(tmp_path, out_path)
            return "downloaded"
        except Exception as e:
            print(f"Direct download of {out_path} failed ({e}), trying gdown...")
        try:
            if gdown.download(id=file_id, output=tmp_path, quiet=True, resume=True):
                os.replace(tmp_path, out_path)
                return "downloaded"
        except Exception as e:
            print(f"Download of {out_path} failed ({e}), retrying...")
        time.sleep(30 * (attempt + 1))
    return "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", help="Output directory")
    args = parser.parse_args()

    session = requests.Session()
    root = {name: (item_id, mime) for item_id, name, mime in list_folder(session, OPTC_ROOT_FOLDER)}

    files = [(root[GROUND_TRUTH_FILE][0], GROUND_TRUTH_FILE)]
    ecar_bro = {name: item_id for item_id, name, _ in list_folder(session, root["ecar-bro"][0])}
    for sub in ECAR_BRO_SUBFOLDERS:
        files.extend(walk(session, ecar_bro[sub], os.path.join("ecar-bro", sub)))
    print(f"{len(files)} files to download")

    status = {"downloaded": 0, "skipped": 0, "failed": 0}
    for i, (file_id, rel_path) in enumerate(files):
        result = download(session, file_id, os.path.join(args.output, rel_path))
        status[result] += 1
        print(f"[{i + 1}/{len(files)}] {result}: {rel_path}", flush=True)
    print(f"Done: {status}")
    if status["failed"]:
        print("Some files failed (e.g. Google Drive quotas): run the script again to resume.")


if __name__ == "__main__":
    main()
