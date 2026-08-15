"""Document AI response -> our figure rows.

Two deliberate choices:

* Block types are **not** hardcoded to a guessed enum. Layout Parser's figure blocks are
  matched by substring, and every block type seen is counted and reported, so the smoke
  test tells us what the API actually emits instead of us asserting it up front.
* A figure block's text goes to `caption_generated`, never `caption_printed`. The printed
  caption is what a citation points at; model-written text only widens the search net.
  figextract keeps that split and this must not be the thing that breaks it.
"""

from collections import Counter

FIGURE_HINTS = ("figure", "image", "chart", "picture", "graphic")


def _oneof(block):
    """-> (kind, type_string, text). Handles whichever block variant is populated."""
    for kind in ("text_block", "table_block", "list_block"):
        sub = getattr(block, kind, None)
        if sub is not None and getattr(block, "_pb", None) is not None:
            if not block._pb.HasField(kind):
                continue
        elif sub is None:
            continue
        return (
            kind,
            getattr(sub, "type_", "") or getattr(sub, "type", ""),
            getattr(sub, "text", ""),
        )
    return ("unknown", "", "")


def flatten(blocks, depth=0):
    """Layout blocks nest (a text block can own child blocks). Yield every one."""
    for b in blocks or []:
        yield b, depth
        kind, _, _ = _oneof(b)
        sub = getattr(b, kind, None)
        yield from flatten(getattr(sub, "blocks", None), depth + 1)


def block_bbox_norm(block):
    """-> [x0,y0,x1,y1] in 0..1, or None when the API gave no geometry for this block."""
    bb = getattr(block, "bounding_box", None)
    if bb is None:
        return None
    verts = list(getattr(bb, "normalized_vertices", []) or [])
    if not verts:
        return None
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    return [round(min(xs), 5), round(min(ys), 5), round(max(xs), 5), round(max(ys), 5)]


def block_page(block):
    """0-based page index within the submitted slice. page_span is 1-based."""
    ps = getattr(block, "page_span", None)
    start = getattr(ps, "page_start", 0) if ps is not None else 0
    return max(0, int(start) - 1)


def survey(document):
    """What did the API actually return? Counts by block type, geometry, and tokens."""
    kinds, typed, boxed, unboxed = Counter(), Counter(), 0, 0
    for b, _ in flatten(document.document_layout.blocks):
        kind, type_, _ = _oneof(b)
        kinds[kind] += 1
        typed[type_ or "(none)"] += 1
        if block_bbox_norm(b):
            boxed += 1
        else:
            unboxed += 1
    tokens = sum(len(getattr(p, "tokens", []) or []) for p in getattr(document, "pages", []))
    return {
        "block_kinds": dict(kinds),
        "block_types": dict(typed),
        "blocks_with_bbox": boxed,
        "blocks_without_bbox": unboxed,
        "pages_in_response": len(getattr(document, "pages", []) or []),
        "token_count": tokens,
    }


def is_figure(type_):
    t = (type_ or "").lower()
    return any(h in t for h in FIGURE_HINTS)


def figure_rows(document, page_ids, sizes):
    """-> rows in figextract's schema.

    `page_ids` maps slice index -> 'pXXXX'; `sizes` maps slice index -> (w, h) in pixels
    at render dpi, so normalized coords become the same image coords figextract wrote.
    """
    rows = []
    seq = Counter()
    for b, _ in flatten(document.document_layout.blocks):
        kind, type_, text = _oneof(b)
        if not is_figure(type_):
            continue
        i = block_page(b)
        if i >= len(page_ids):
            continue
        bn = block_bbox_norm(b)
        if bn is None:
            continue
        w, h = sizes[i]
        page_id = page_ids[i]
        seq[page_id] += 1
        rows.append(
            {
                "page_id": page_id,
                "slice_index": i,
                "cls": type_ or kind,
                "score": None,  # Layout Parser returns no confidence per block
                "bbox_norm": bn,
                "bbox_img": [int(bn[0] * w), int(bn[1] * h), int(bn[2] * w), int(bn[3] * h)],
                "block_id": getattr(b, "block_id", ""),
                "caption_printed": None,  # this API does not separate a printed caption
                "caption_generated": (text or "").strip() or None,
                "caption_source": "docai:layout_parser",
                "fig_label": None,
                "detector": "docai_layout",
                "impl": "documentai/layout_parser",
                "_seq": seq[page_id],
            }
        )
    return rows
