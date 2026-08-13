# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TrOCR benchmark
#
# This notebook runs TrOCR on all 24 committed held-out Pierce pages (`p0024`–`p0047`).
# It uses the existing DocLayout-YOLO non-figure regions, splits each text region into
# line crops, and sends those line crops to TrOCR. It does not rerun layout detection
# and does not use Chandra or Document AI text as OCR input.
#
# The default is `microsoft/trocr-large-printed`, the largest official Microsoft
# printed TrOCR checkpoint. This is a TrOCR-family comparison point, not a claim that
# TrOCR is current OCR state of the art. Use a Tesla T4 GPU.
#
# Pages p0041 and p0043 contain image descriptions in the committed labels. The
# notebook reports both all-page metrics and a text-only subset that excludes those
# two semantic image-description labels.

# %%
# Kaggle: Internet ON, Accelerator = Tesla T4. Do not select P100.
# %pip install -q 'transformers>=4.46,<5' 'sentencepiece>=0.2,<1' 'safetensors>=0.4' 'pillow>=10,<12' 'opencv-python-headless>=4.10,<5'

from pathlib import Path
import subprocess
import sys

REPO = Path('/kaggle/working/doc-agent-G07')
if not (REPO / 'grading_kit/labels.jsonl').is_file():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'main',
                    'https://github.com/smammahdi/doc-agent-G07.git', str(REPO)],
                   check=True)

HELDOUT = REPO / 'grading_kit/heldout_pages'
LABELS = REPO / 'grading_kit/labels.jsonl'
LAYOUT = REPO / 'extras/output/doclayout-yolo/detections.jsonl'
assert LABELS.is_file(), LABELS
assert LAYOUT.is_file(), LAYOUT
PAGES = [f'p{i:04d}' for i in range(24, 48)]
assert all((HELDOUT / f'{pid}.jpg').is_file() for pid in PAGES)
print({'heldout_pages': len(PAGES), 'layout_file': str(LAYOUT)})

# %%
import json
import re
import time
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

if not torch.cuda.is_available():
    raise RuntimeError('Select a Tesla T4 GPU before running this notebook')
print({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__})

MODEL_NAME = 'microsoft/trocr-large-printed'
BATCH_SIZE = 8
MAX_LENGTH = 128
OUT = Path('/kaggle/working/trocr-large-printed-heldout')
OUT.mkdir(parents=True, exist_ok=True)

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, use_safetensors=True
).eval().cuda()
print({'model': MODEL_NAME, 'device': 'cuda', 'dtype': 'float16'})


# %%
def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

labels = {row['page_id']: row['text'] for row in read_jsonl(LABELS)}
regions_by_page = {pid: [] for pid in PAGES}
for row in read_jsonl(LAYOUT):
    pid = row['page_id']
    if pid in regions_by_page and not row.get('is_figure', False):
        regions_by_page[pid].append(row)
for pid in PAGES:
    regions_by_page[pid].sort(key=lambda row: (row['bbox_norm'][1], row['bbox_norm'][0]))
print('non-figure regions:', sum(len(rows) for rows in regions_by_page.values()))

def bbox_pixels(box, width, height):
    x0, y0, x1, y1 = box
    left = max(0, min(width, int(x0 * width)))
    top = max(0, min(height, int(y0 * height)))
    right = max(left + 1, min(width, int(x1 * width + 0.999999)))
    bottom = max(top + 1, min(height, int(y1 * height + 0.999999)))
    return left, top, right, bottom

def split_lines(crop):
    gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    row_ink = np.count_nonzero(binary, axis=1)
    active = row_ink >= max(1, int(round(gray.shape[1] * 0.01)))
    spans = []
    start = None
    gap = 0
    for y, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 3:
                if y - gap - start >= 6:
                    spans.append((start, y - gap + 1))
                start, gap = None, 0
    if start is not None and gray.shape[0] - start >= 6:
        spans.append((start, gray.shape[0]))
    return spans or [(0, gray.shape[0])]

def make_line_crops(image, region):
    left, top, right, bottom = bbox_pixels(region['bbox_norm'], image.width, image.height)
    padding = 4
    left, top = max(0, left - padding), max(0, top - padding)
    right, bottom = min(image.width, right + padding), min(image.height, bottom + padding)
    region_crop = image.crop((left, top, right, bottom)).convert('RGB')
    rows = []
    for line_index, (line_top, line_bottom) in enumerate(split_lines(region_crop)):
        line_image = region_crop.crop((0, line_top, region_crop.width, line_bottom))
        rows.append((line_index, line_image, (left, top + line_top, right, top + line_bottom)))
    return rows



# %%
@torch.inference_mode()
def recognize(images):
    values = processor(images=images, return_tensors='pt').pixel_values.cuda().half()
    ids = model.generate(values, max_length=MAX_LENGTH, num_beams=1)
    return [text.strip() for text in processor.batch_decode(ids, skip_special_tokens=True)]

for pid in PAGES:
    target = OUT / f'{pid}.json'
    if target.exists():
        continue
    started = time.perf_counter()
    image = Image.open(HELDOUT / f'{pid}.jpg').convert('RGB')
    line_items = []
    for region_index, region in enumerate(regions_by_page[pid]):
        for line_index, crop, bbox_px in make_line_crops(image, region):
            line_items.append((region_index, line_index, region, crop, bbox_px))
    recognized = []
    for start in range(0, len(line_items), BATCH_SIZE):
        batch = line_items[start:start + BATCH_SIZE]
        recognized.extend(recognize([item[3] for item in batch]))
    lines = []
    for item, text in zip(line_items, recognized):
        region_index, line_index, region, _, bbox_px = item
        lines.append({
            'line_id': f'{pid}:r{region_index:04d}:l{line_index:03d}',
            'region_index': region_index,
            'source_class': region['class_name'],
            'bbox_norm': region['bbox_norm'],
            'bbox_px': list(bbox_px),
            'text': text,
        })
    record = {
        'page_id': pid,
        'model': MODEL_NAME,
        'layout': 'doclayout_yolo',
        'status': 'complete',
        'region_count': len(regions_by_page[pid]),
        'line_count': len(lines),
        'text': '\n'.join(line['text'] for line in lines if line['text']),
        'lines': lines,
        'elapsed_seconds': round(time.perf_counter() - started, 3),
    }
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(pid, 'regions=', record['region_count'], 'lines=', record['line_count'], 'seconds=', record['elapsed_seconds'])
print('OCR complete:', len(list(OUT.glob('p*.json'))), 'pages')


# %%
def normalize(text):
    return re.sub(r'\s+', ' ', text).strip()

def levenshtein(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, value in enumerate(left, 1):
        current = [index]
        for other_index, other in enumerate(right, 1):
            current.append(min(previous[other_index] + 1, current[-1] + 1,
                                previous[other_index - 1] + (value != other)))
        previous = current
    return previous[-1]

def word_f1(hypothesis, reference):
    hyp = Counter(normalize(hypothesis).split())
    ref = Counter(normalize(reference).split())
    true_positive = sum((hyp & ref).values())
    if not hyp and not ref: return 1.0
    if not true_positive: return 0.0
    precision = true_positive / sum(hyp.values())
    recall = true_positive / sum(ref.values())
    return 2 * precision * recall / (precision + recall)

def score(page_ids):
    rows = []
    total_ce = total_we = total_chars = total_words = 0
    for pid in page_ids:
        reference = normalize(labels[pid])
        hypothesis = normalize(json.loads((OUT / f'{pid}.json').read_text())['text'])
        reference_words, hypothesis_words = reference.split(), hypothesis.split()
        ce = levenshtein(hypothesis, reference)
        we = levenshtein(hypothesis_words, reference_words)
        row = {'page_id': pid, 'cer': ce / max(1, len(reference)),
               'wer': we / max(1, len(reference_words)),
               'word_f1': word_f1(hypothesis, reference),
               'reference_chars': len(reference), 'reference_words': len(reference_words)}
        rows.append(row)
        total_ce += ce; total_we += we; total_chars += len(reference); total_words += len(reference_words)
    return {'pages': len(rows), 'micro_cer': total_ce / max(1, total_chars),
            'micro_wer': total_we / max(1, total_words),
            'macro_cer': sum(row['cer'] for row in rows) / len(rows),
            'macro_wer': sum(row['wer'] for row in rows) / len(rows),
            'macro_word_f1': sum(row['word_f1'] for row in rows) / len(rows),
            'per_page': rows}

all_metrics = score(PAGES)
text_only_pages = [pid for pid in PAGES if pid not in {'p0041', 'p0043'}]
text_only_metrics = score(text_only_pages)
metrics = {'engine': MODEL_NAME, 'layout': 'DocLayout-YOLO non-figure regions',
           'all_pages': all_metrics, 'text_only_pages': text_only_metrics,
           'label_note': 'p0041 and p0043 include image descriptions; text-only excludes them'}
(OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: metrics[key] for key in ('engine', 'all_pages', 'text_only_pages') if key != 'engine'}, indent=2))

# %%
import shutil
archive = shutil.make_archive('/kaggle/working/trocr-large-printed-heldout-results', 'zip', OUT)
print('Download:', archive)
