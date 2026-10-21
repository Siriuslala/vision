"""Diagnostic point metric "hit under either coordinate convention" (docs/analysis/plan.md, section 6 item 9).

The strict score reads a point answer (x, y) only as 0-1000 normalized coordinates.  Qwen models
sometimes answer in absolute pixels instead, so the same output is also read as pixels:

    h_norm = [(x W / 1000, y H / 1000) in B]
    h_pix  = [x < W, y < H, (x, y) in B]
    h_any  = max(h_norm, h_pix)

where W x H is the canvas and B the target region (the target's grid cell, half-open).  Because
taking the union of two readings raises the chance level, `chance_rates` gives the hit probability
of a point drawn uniformly from the output square [0, 1000]^2 under each reading.

The helpers never modify strict scores; callers store their output as additional columns.
"""
from __future__ import annotations

import math

from analysis_tools.behavior.decomposition import answer_part, parse_json

OUTPUT_RANGE = 1000.0


def _is_num(v) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def parse_point(response: str) -> tuple[float, float] | None:
    """Point (x, y) from a raw answer, accepting the same wrappers as the content-diagnostic score.

    Accepted: [{"point_2d": [x, y]}], {"point_2d": [x, y]} and a bare [x, y].  Values are not
    range-checked, because an answer in pixels may exceed 1000 on canvases wider than 1000 pixels.
    Returns None for anything else (the caller keeps such answers in the denominator).
    """
    try:
        value = parse_json(answer_part(response))
    except ValueError:
        return None
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, list) and len(value) == 2 and all(_is_num(v) for v in value):
        value = [{'point_2d': value}]
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict)):
        return None
    point = value[0].get('point_2d')
    if not (isinstance(point, list) and len(point) == 2 and all(_is_num(v) for v in point)):
        return None
    return float(point[0]), float(point[1])


def _inside(p: tuple[float, float], box) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= p[0] < x1 and y0 <= p[1] < y1


def convention_hits(point: tuple[float, float] | None, canvas, box) -> dict:
    """Hit indicators of one point answer under both conventions.

    point:  (x, y) as written by the model, or None for an unusable answer (all hits False).
    canvas: (W, H) in pixels.  box: target region (x0, y0, x1, y1) in pixels, half-open.
    """
    if point is None:
        return {'point_valid': False, 'h_norm': False, 'h_pix': False, 'h_any': False, 'pixel_only': False}
    W, H = canvas
    x, y = point
    h_norm = 0 <= x <= OUTPUT_RANGE and 0 <= y <= OUTPUT_RANGE and _inside((x * W / OUTPUT_RANGE, y * H / OUTPUT_RANGE), box)
    h_pix = 0 <= x < W and 0 <= y < H and _inside((x, y), box)
    return {'point_valid': True, 'h_norm': bool(h_norm), 'h_pix': bool(h_pix), 'h_any': bool(h_norm or h_pix),
            'pixel_only': bool(h_pix and not h_norm)}


def _overlap(a, b) -> float:
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def chance_rates(canvas, box) -> dict:
    """Hit probabilities of a point uniform on [0, 1000]^2 under each reading and their union.

    In output space the normalized reading hits the box B scaled by (1000/W, 1000/H); the pixel
    reading hits B itself clipped to the output square.  p_any = p_norm + p_pix - p_both.
    """
    W, H = canvas
    x0, y0, x1, y1 = box
    square = (0., 0., OUTPUT_RANGE, OUTPUT_RANGE)
    region_norm = (x0 * OUTPUT_RANGE / W, y0 * OUTPUT_RANGE / H, x1 * OUTPUT_RANGE / W, y1 * OUTPUT_RANGE / H)
    region_pix = (x0, y0, x1, y1)
    area = OUTPUT_RANGE ** 2
    p_norm = _overlap(region_norm, square) / area
    p_pix = _overlap(region_pix, square) / area
    both = (max(region_norm[0], region_pix[0]), max(region_norm[1], region_pix[1]),
            min(region_norm[2], region_pix[2]), min(region_norm[3], region_pix[3]))
    p_both = _overlap(both, square) if both[0] < both[2] and both[1] < both[3] else 0.
    p_both /= area
    return {'p_norm': p_norm, 'p_pix': p_pix, 'p_any': p_norm + p_pix - p_both}


def score_point(response: str, finish_reason: str, canvas, box) -> dict:
    """Convention diagnostic of one generated point answer; truncated answers count as misses."""
    point = parse_point(response) if finish_reason == 'eos' else None
    out = convention_hits(point, canvas, box)
    out['point'] = list(point) if point is not None else None
    out.update(chance_rates(canvas, box))
    return out
