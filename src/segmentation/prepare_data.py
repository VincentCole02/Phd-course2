"""Prepare a small ISIC2017 Task 1 dataset from official public sources."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import random
import re
import threading
from zipfile import ZipFile

from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


MASK_BASE = "https://isic-archive.s3.amazonaws.com/challenges/2017/"
IMAGE_BASE = "https://isic-archive.s3.us-east-1.amazonaws.com/"
SPLITS = {
    "train": ("Training", 2000),
    "val": ("Validation", 150),
    "test": ("Test_v2", 600),
}
FIELDS = ("split", "isic_id", "image_source", "mask_source")
MASK_NAME = re.compile(r"(ISIC_\d{7})_segmentation\.png", re.IGNORECASE)
LOCAL = threading.local()


def fetch(url, allow_missing=False):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        retry = Retry(total=4, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        LOCAL.session.mount("https://", HTTPAdapter(max_retries=retry))
    with LOCAL.session.get(url, timeout=(15, 45)) as response:
        if allow_missing and response.status_code in (403, 404):
            return None
        response.raise_for_status()
        return response.content


def write_bytes(path, content):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def prepare_pair(directory, split, isic_id, mask_url, mask_bytes):
    thumb_url = f"{IMAGE_BASE}thumbnails/{isic_id}_thumbnail.jpg"
    full_url = f"{IMAGE_BASE}images/{isic_id}.jpg"
    image_path = directory / f"{isic_id}.jpg"
    full_path = directory / f"{isic_id}.full.jpg"
    image_source = thumb_url
    if full_path.is_file():
        image_source = full_url
    elif not image_path.is_file():
        image = fetch(thumb_url, allow_missing=True)
        if image is None:
            image = fetch(full_url)
            image_path = full_path
            image_source = full_url
        with Image.open(BytesIO(image)) as decoded:
            if decoded.format != "JPEG":
                raise ValueError(f"Expected JPEG for {isic_id}")
            decoded.verify()
        write_bytes(image_path, image)

    mask_path = directory / f"{isic_id}_Segmentation.png"
    if not mask_path.is_file():
        with Image.open(BytesIO(mask_bytes)) as mask:
            mask = mask.convert("L").resize((256, 256), Image.Resampling.NEAREST)
            mask = mask.point(lambda value: 255 if value >= 128 else 0)
            output = BytesIO()
            mask.save(output, format="PNG", optimize=True)
        write_bytes(mask_path, output.getvalue())
    return split, isic_id, image_source, mask_url


def prepare(data_dir, train_count, seed, workers):
    cache = data_dir / ".download-cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, (archive_split, expected_count) in SPLITS.items():
        name = f"ISIC-2017_{archive_split}_Part1_GroundTruth.zip"
        mask_url = MASK_BASE + name
        archive_path = cache / name
        if not archive_path.is_file():
            write_bytes(archive_path, fetch(mask_url))
        directory = data_dir / split
        directory.mkdir(exist_ok=True)
        with ZipFile(archive_path) as archive, ThreadPoolExecutor(max_workers=workers) as pool:
            members = {}
            for member in archive.namelist():
                match = MASK_NAME.fullmatch(Path(member).name)
                if match:
                    isic_id = match[1]
                    if isic_id in members:
                        raise ValueError(f"Duplicate annotation for {isic_id} in {name}")
                    members[isic_id] = member
            if len(members) != expected_count:
                raise ValueError(f"Expected {expected_count} masks in {name}, got {len(members)}")
            ids = sorted(members)
            if split == "train":
                ids = sorted(random.Random(seed).sample(ids, train_count))
            print(f"{split}: preparing {len(ids)} image/mask pairs", flush=True)
            jobs = [
                pool.submit(prepare_pair, directory, split, isic_id, mask_url, archive.read(members[isic_id]))
                for isic_id in ids
            ]
            for index, job in enumerate(jobs, 1):
                rows.append(job.result())
                if index % 100 == 0 or index == len(ids):
                    print(f"  {index}/{len(ids)}", flush=True)

    ids = [row[1] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("ISIC image IDs overlap across official splits")
    manifest = data_dir / "manifest.csv"
    temporary = manifest.with_suffix(".csv.part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    temporary.replace(manifest)
    print(f"Ready: {len(rows)} image/mask pairs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/isic2017"))
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.train_count <= 2000 or args.workers < 1:
        parser.error("--train-count must be in [1, 2000] and --workers must be positive")
    prepare(args.data, args.train_count, args.seed, args.workers)
