import gdown
import os
import argparse
from google_drive_urls import url_map

def main(args):
    host = args.host
    base_dir = args.output
    os.makedirs(base_dir, exist_ok=True)

    start_ind = args.index
    mapping = url_map[host]
    num = len((mapping))
    for i in range(num):
        if i >= start_ind:
            relative_dir, url = mapping[i]
            out_dir = os.path.join(base_dir, relative_dir)
            print(f"Start downloading {i}/{num}-th folder to {out_dir}")
            print(f"URL: {url}")

            gdown.download_folder(url=url, output=out_dir, quiet=False, resume=True)

            print(f"Finish downloading {i}/{num}-th folder to {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('host', help='The number of host.')
    parser.add_argument('output', help='Output base dir.')
    args = parser.parse_args()

    main(args)