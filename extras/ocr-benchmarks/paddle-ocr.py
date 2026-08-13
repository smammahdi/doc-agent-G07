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
# # PaddleOCR benchmark
#
# This is a small, controlled OCR check on the 24 committed held-out Pierce pages (`p0024`–`p0047`). It uses the page images and `grading_kit/labels.jsonl` already in this repository. It runs PaddleOCR's current default PP-OCRv6 English pipeline directly on the full page, saves text and line-box JSON, and reports CER, WER, and word-F1.
#
# This is an OCR benchmark only. It does not rerun Chandra, DocLayout-YOLO, or any other layout model.

# %%
# Kaggle: Internet ON. GPU is optional for this held-out check.
# %pip install -q 'paddleocr==3.7.0'
# %pip install -q 'paddlepaddle-gpu==3.3.1' -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
# Kaggle may ship torch with older CUDA libraries. PaddleX imports torch indirectly,
# so align the three libraries required by Kaggle's torch 2.10 CUDA build.
# %pip install -q --upgrade --force-reinstall --no-deps \
#     'nvidia-nccl-cu12==2.27.5' \
#     'nvidia-nvjitlink-cu12==12.8.93' \
#     'nvidia-nvtx-cu12==12.8.90'
# %pip install -q 'pymupdf>=1.25,<1.26' 'pillow>=10,<12'

import os
os.environ['PADDLE_PDX_CACHE_HOME'] = '/kaggle/working/paddlex-cache'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import paddle
import paddleocr
print({'paddle': paddle.__version__, 'paddleocr': paddleocr.__version__,
       'device': paddle.get_device(), 'cuda_build': paddle.is_compiled_with_cuda()})

# %%
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
assert LABELS.is_file(), LABELS
assert all((HELDOUT / f'p{i:04d}.jpg').is_file() for i in range(24, 48))
print('held-out pages:', len(list(HELDOUT.glob('p*.jpg'))))

# %%
import json
import time
from collections import Counter
from paddleocr import PaddleOCR

OUT = Path('/kaggle/working/paddleocr-heldout')
OUT.mkdir(parents=True, exist_ok=True)
labels = {json.loads(line)['page_id']: json.loads(line)['text']
          for line in LABELS.read_text(encoding='utf-8').splitlines() if line.strip()}
pages = sorted(labels)
assert pages == [f'p{i:04d}' for i in range(24, 48)], pages

# PaddleOCR 3.7's English pipeline defaults to its PP-OCRv6 detector/recognizer.
ocr = PaddleOCR(lang='en', use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_textline_orientation=False,
                device='gpu:0' if paddle.is_compiled_with_cuda() else 'cpu')

def result_dict(value):
    payload = getattr(value, 'json', None)
    if callable(payload): payload = payload()
    if isinstance(payload, str): payload = json.loads(payload)
    return payload if isinstance(payload, dict) else value if isinstance(value, dict) else {}

def lines_from(result):
    data = result_dict(result)
    data = data.get('res', data)
    texts, scores = data.get('rec_texts', []), data.get('rec_scores', [])
    return [{'text': str(t), 'score': float(scores[i]) if i < len(scores) else None}
            for i, t in enumerate(texts) if str(t).strip()]

for pid in pages:
    target = OUT / f'{pid}.json'
    if target.exists():
        continue
    started = time.perf_counter()
    result = list(ocr.predict(str(HELDOUT / f'{pid}.jpg')))
    lines = []
    for item in result: lines.extend(lines_from(item))
    record = {'page_id': pid, 'text': '\n'.join(x['text'] for x in lines),
              'lines': lines, 'elapsed_seconds': round(time.perf_counter()-started, 3)}
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(pid, len(lines), record['elapsed_seconds'])
print('OCR complete:', len(list(OUT.glob('p*.json'))), 'pages')

# %%
import re

def norm(text): return re.sub(r'\s+', ' ', text).strip()
def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]
def word_f1(hyp, ref):
    h, r = Counter(norm(hyp).split()), Counter(norm(ref).split())
    tp = sum((h & r).values())
    if not h and not r: return 1.0
    if not tp: return 0.0
    precision, recall = tp / sum(h.values()), tp / sum(r.values())
    return 2 * precision * recall / (precision + recall)

rows, total_ce, total_we, total_chars, total_words = [], 0, 0, 0, 0
for pid in pages:
    ref = norm(labels[pid]); hyp = norm(json.loads((OUT / f'{pid}.json').read_text())['text'])
    ref_words, hyp_words = ref.split(), hyp.split()
    ce, we = lev(hyp, ref), lev(hyp_words, ref_words)
    row = {'page_id': pid, 'cer': ce / max(1, len(ref)), 'wer': we / max(1, len(ref_words)),
           'word_f1': word_f1(hyp, ref), 'reference_chars': len(ref), 'reference_words': len(ref_words)}
    rows.append(row); total_ce += ce; total_we += we; total_chars += len(ref); total_words += len(ref_words)
    print(pid, row)
metrics = {'engine': 'PaddleOCR 3.7.0 PP-OCRv6 default pipeline',
           'layout': 'full-page PaddleOCR detection', 'pages': len(rows),
           'micro_cer': total_ce / max(1, total_chars), 'micro_wer': total_we / max(1, total_words),
           'macro_cer': sum(r['cer'] for r in rows) / len(rows),
           'macro_wer': sum(r['wer'] for r in rows) / len(rows),
           'macro_word_f1': sum(r['word_f1'] for r in rows) / len(rows), 'per_page': rows}
(OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
print(json.dumps({k: metrics[k] for k in metrics if k != 'per_page'}, indent=2))

# %%
import shutil
archive = shutil.make_archive('/kaggle/working/paddleocr-heldout-results', 'zip', OUT)
print('Download from Kaggle output:', archive)
