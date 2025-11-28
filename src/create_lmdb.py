import argparse
import pickle

import cv2
import lmdb
from path import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=Path, required=True)
args = parser.parse_args()

data_dir = args.data_dir
words_txt = data_dir / 'words.txt'
img_root = data_dir / 'img'

if not words_txt.exists():
    raise FileNotFoundError(f'{words_txt} not found')

if not img_root.exists():
    raise FileNotFoundError(f'{img_root} not found (expected images here)')

# build index of all png files under img/ by basename
fn_imgs = list(img_root.walkfiles('*.png'))
png_index = {p.basename(): p for p in fn_imgs}

# 2GB is enough for IAM dataset
assert not (data_dir / 'lmdb').exists()
env = lmdb.open(str(data_dir / 'lmdb'), map_size=1024 * 1024 * 1024 * 2)

added = 0
missing = 0

# and put only imgs referenced in words.txt into lmdb as pickled grayscale imgs
with words_txt.open('r', encoding='utf-8') as f, env.begin(write=True) as txn:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        img_id = parts[0]              # e.g. a01-000u-00-00
        basename = img_id + '.png'     # expected filename

        fn_img = png_index.get(basename)
        if fn_img is None:
            missing += 1
            continue

        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            missing += 1
            continue

        txn.put(basename.encode("ascii"), pickle.dumps(img))
        added += 1

print(f'Added {added} images to LMDB')
if missing:
    print(f'Skipped {missing} entries (image file not found or unreadable)')

env.close()