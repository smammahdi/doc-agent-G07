G07 Pierce layout bounding-box visual verification bundle

Pages: p0024-p0047 (the 24 currently committed held-out pages).
Each PDF contains the original scanned page with vector boxes overlaid.

Color legend for layout outputs:
  green = text/plain text
  orange = caption/title/footnote
  red = image/figure/diagram/chart
  blue = header/heading/number/title
  purple = table
  cyan = Document AI word boxes (reference only)

Included:
  chandra_heldout_bbox.pdf: Chandra blocks normalized per row page_box
  projection_heldout_bbox.pdf: starter row-projection heuristic, generated for these pages
  orphan_ink_heldout_bbox.pdf: full-book 150-DPI heuristic rerun
  doclayout_yolo_heldout_bbox.pdf: full-book 150-DPI rerun
  ppdoclayout_v3_heldout_bbox.pdf: full-book 150-DPI rerun
  ppdoclayout_plus_l_heldout_bbox.pdf: prior full-book PaddleX benchmark output
  picodet_s_heldout_bbox.pdf: prior full-book PaddleX benchmark output
  document_ai_word_boxes_heldout_bbox.pdf: Document AI OCR word boxes, not a layout model

PaddleOCR is an OCR engine, not a layout detector, so it is not included in this layout bundle.
Not included as model outputs:
  fused consensus: it combines detectors and is not an independent model
  Eynollah: existing output is empty
  Surya: not run

All files are for visual inspection only. Chandra and Document AI are silver/reference
artifacts, not human ground truth. The plus-L and PicoDet files came from the prior
full-book benchmark workspace and were not added to the Git repository.
