"""Builds the photo zips for the multimodal evaluation assignment from NaturalBench.

Usage: python3 make_test_images.py PARQUET_DIR

PARQUET_DIR holds the three NaturalBench shards (data/train-0000{0,1,2}-of-00003.parquet from
https://huggingface.co/datasets/BaiqiL/NaturalBench). Every pair below was checked by eye.

Writes into Assignments/2_multimodal_+_advanced_models/:
  multimodal_eval_images.zip            the 30-item class test set (students download this)
  multimodal_eval_solution_images.zip   the 4 extra pairs used by the solution's MY_ITEMS
and prints the item lists that build_assignment.py hard-codes.
"""
import hashlib
import io
import os
import sys
import zipfile

import pyarrow.parquet as pq
from PIL import Image

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = "multimodal_eval_images"
MAX_SIDE = 640

# (NaturalBench index, category, question, choices, answer for Image_0, answer for Image_1)
TEST_PAIRS = [
    (104, "count", "How many apples are in the image: two or eight?", ["2", "8"], "2", "8"),
    (1791, "count", "How many eggs are in the image: one or two?", ["1", "2"], "1", "2"),
    (1439, "count", "How many cats are in the image: one or two?", ["1", "2"], "2", "1"),
    (1650, "count", "How many cats are in the image: one or two?", ["1", "2"], "1", "2"),
    (399, "count", "How many traffic cones are in the image: one or two?", ["1", "2"], "1", "2"),
    (1563, "count", "How many horses are in the image: two or eight?", ["2", "8"], "2", "8"),
    (383, "color", "What color is the car: red or orange?", ["red", "orange"], "red", "orange"),
    (1691, "color", "What color is the truck: white or black?", ["white", "black"], "white", "black"),
    (1808, "color", "What color is the bike: black or purple?", ["black", "purple"], "black", "purple"),
    (480, "color", "What color is the sign: red or green?", ["red", "green"], "red", "green"),
    (1869, "color", "What color is the table: orange or red?", ["orange", "red"], "orange", "red"),
    (485, "color", "What color are the church's front doors: white or black?", ["white", "black"], "white", "black"),
    (10, "direction", "Is the heron facing left or right?", ["left", "right"], "right", "left"),
    (867, "direction", "Is the airplane facing left or right?", ["left", "right"], "right", "left"),
    (326, "direction", "Is the car facing left or right?", ["left", "right"], "right", "left"),
    (842, "direction", "Is the lizard facing left or right?", ["left", "right"], "left", "right"),
    (1259, "direction", "Is the grasshopper facing left or right?", ["left", "right"], "left", "right"),
    (1762, "direction", "Is the van facing to the left?", ["yes", "no"], "yes", "no"),
    (665, "state", "Is the tiger standing or sitting?", ["standing", "sitting"], "standing", "sitting"),
    (961, "state", "Is the dog sitting or lying down?", ["sitting", "lying"], "lying", "sitting"),
    (24, "state", "Is the deer standing or lying down?", ["standing", "lying"], "standing", "lying"),
    (1292, "state", "Is the light turned on?", ["yes", "no"], "no", "yes"),
    (443, "state", "Was this photo taken indoors or outdoors?", ["indoors", "outdoors"], "indoors", "outdoors"),
    (316, "state", "Are the butterfly's wings open or closed?", ["open", "closed"], "open", "closed"),
    (53, "text", 'Does the wall say "NO DUMPING"?', ["yes", "no"], "yes", "no"),
    (157, "text", 'Is there a sign that says "EXIT"?', ["yes", "no"], "no", "yes"),
    (745, "text", 'Is the word "WATER" shown in the image?', ["yes", "no"], "yes", "no"),
    (15, "text", "Which word is on the sign: restore or winter?", ["restore", "winter"], "restore", "winter"),
    (318, "text", 'Is there a red "EXIT" sign in the image?', ["yes", "no"], "yes", "no"),
    (396, "text", 'Does the manhole cover say "WATER"?', ["yes", "no"], "no", "yes"),
]

SOLUTION_PAIRS = [
    (715, "material", "What is the wall made of: brick or wood?", ["brick", "wood"], "brick", "wood"),
    (853, "material", "What are the bench legs made of: metal or wood?", ["metal", "wood"], "metal", "wood"),
    (293, "material", "What is the star-shaped object made of: metal or stone?", ["metal", "stone"], "metal", "stone"),
    (240, "material", "Is the wall made of brick or stucco?", ["brick", "stucco"], "brick", "stucco"),
]

SWAP = {867, 853}

CREDITS = """Test images for the F.O.R.G.E. AI Challenge assignment "Evaluate a Multimodal Model".

The photos and the paired questions come from NaturalBench (Li et al., NeurIPS 2024),
https://huggingface.co/datasets/BaiqiL/NaturalBench, released under the Apache 2.0 license.
The photos themselves come from DOCCI (Onoe et al., 2024),
https://huggingface.co/datasets/google/docci, released under CC BY 4.0.

The photos were resized to at most 640 pixels on a side. Some questions were reworded.
NaturalBench indices used, in item order: {indices}
"""


def load_rows(parquet_dir, wanted):
    rows = {}
    for name in sorted(os.listdir(parquet_dir)):
        if name.endswith(".parquet"):
            table = pq.read_table(os.path.join(parquet_dir, name), columns=["Index", "Image_0", "Image_1"])
            for r in table.to_pylist():
                if r["Index"] in wanted:
                    rows[r["Index"]] = r
    missing = wanted - set(rows)
    assert not missing, f"indices not found: {sorted(missing)}"
    return rows


def jpeg_bytes(raw):
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def build(pairs, rows, zip_name, credits):
    items, seen, counts = [], set(), {}
    with zipfile.ZipFile(os.path.join(OUT, zip_name), "w", zipfile.ZIP_DEFLATED) as z:
        for idx, category, question, choices, ans0, ans1 in pairs:
            n = counts[category] = counts.get(category, -1) + 1
            item_id = f"{category}-{n}"
            images = [(rows[idx]["Image_0"]["bytes"], ans0), (rows[idx]["Image_1"]["bytes"], ans1)]
            if idx in SWAP:                # use Image_1 as the original, to balance the answers
                images.reverse()
            (orig, answer), (cf, cf_answer) = images
            for raw, fname in [(orig, f"{item_id}.jpg"), (cf, f"{item_id}_cf.jpg")]:
                data = jpeg_bytes(raw)
                digest = hashlib.sha1(data).hexdigest()
                assert digest not in seen, f"duplicate photo in {item_id}"
                seen.add(digest)
                z.writestr(f"{FOLDER}/{fname}", data)
            assert answer != cf_answer and answer in choices and cf_answer in choices
            items.append({"id": item_id, "category": category, "question": question, "answer": answer,
                          "choices": choices, "image": f"{item_id}.jpg", "cf_image": f"{item_id}_cf.jpg",
                          "cf_answer": cf_answer})
        if credits:
            z.writestr(f"{FOLDER}/CREDITS.txt", CREDITS.format(indices=", ".join(str(p[0]) for p in pairs)))
    print(f"wrote {zip_name} ({os.path.getsize(os.path.join(OUT, zip_name)) / 1e6:.1f} MB)")
    return items


def as_literal(items):
    return "[\n" + "".join(f"    {item!r},\n" for item in items) + "]"


if __name__ == "__main__":
    rows = load_rows(sys.argv[1], {p[0] for p in TEST_PAIRS + SOLUTION_PAIRS})
    test_items = build(TEST_PAIRS, rows, "multimodal_eval_images.zip", credits=True)
    solution_items = build(SOLUTION_PAIRS, rows, "multimodal_eval_solution_images.zip", credits=False)
    print("TEST_SET =", as_literal(test_items))
    print("MY_ITEMS =", as_literal(solution_items))
