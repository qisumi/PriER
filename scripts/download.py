#!/usr/bin/env python3
"""Download the raw benchmark archives (small public research datasets)."""
import argparse
import hashlib
import urllib.request
from pathlib import Path

FILES = {
    # Leipzig benchmark group (CC-BY-4.0)
    "Abt-Buy.zip": "https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip",
    "DBLP-ACM.zip": "https://dbs.uni-leipzig.de/files/datasets/DBLP-ACM.zip",
    "affiliationstrings.zip": "https://dbs.uni-leipzig.de/files/datasets/affiliationstrings.zip",
    "musicbrainz-20-A01.csv.dapo": "https://dbs.uni-leipzig.de/files/datasets/saeedi/musicbrainz-20-A01.csv.dapo",
    "Amazon-GoogleProducts.zip": "https://dbs.uni-leipzig.de/files/datasets/Amazon-GoogleProducts.zip",
    # Restaurant (CC0)
    "restaurant.RData": "https://raw.githubusercontent.com/cleanzr/restaurant/a38c037250a069c4aaf30be9c7e0234e8d560933/data/restaurant.RData",
    # Walmart-Amazon transfer benchmark (deepmatcher data page)
    "walmart_amazon_raw_data.zip": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/walmart_amazon_raw_data.zip",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data")
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = dest / name
        if target.exists():
            print(f"skip {name}")
            continue
        print(f"downloading {name} ...")
        with urllib.request.urlopen(url) as response, open(target, "wb") as out:
            out.write(response.read())
        print(f"  {target.stat().st_size} bytes, sha256 "
              f"{hashlib.sha256(target.read_bytes()).hexdigest()[:16]}...")


if __name__ == "__main__":
    main()
