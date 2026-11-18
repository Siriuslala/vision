"""Experiment 7.4: located mixed-data fine-tuning for BabyVision row/column find-the-difference (model-free part).

Results doc: docs/analysis/application/results_2026-11-11_babyvision_rowcol.md.

Contents
* BabyVision-style replica grids (single image, one odd entity, row/column answer) rendered from scratch: no
  BabyVision image or crop is used. Families differ in entity content; every family has a training content set
  and a disjoint evaluation content set (letter pairs, word lists, templates, group members), and the 'number'
  family is evaluation-only (held-out family). Grid sizes and cell sizes follow the 16 annotated BabyVision grids:
  7-28 entities per axis, cells of 0.6-3 merged tokens (19-94 pixels).
* Prompts: direct row/column JSON, direct point, BabyVision-style reasoning prompt with "Final answer: (r,c)",
  each for the odd entity and for a recoloured (marked) entity.
* The coordinate-mediated reasoning answer used in training (target point, grid extent, division by the stated
  row / column counts) and its exact arithmetic check.
* Error classes of the diagnosis (detection / ordinal / format) and the 7.4 adapter conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import math
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analysis_tools.common import stimuli as st

FONT_DIR = '/usr/share/fonts/truetype/dejavu/'   # system font shipped with the OS image (not a machine path)
FONTS = ('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSerif.ttf', 'DejaVuSansMono.ttf', 'DejaVuSansCondensed.ttf')
FOREGROUND_THRESHOLD = 60
FAMILIES = ('letter', 'word', 'number', 'shape', 'group')
TRAIN_FAMILIES = ('letter', 'word', 'shape', 'group')          # 'number' is held out from training entirely

# Content sets: (common, odd). Training and evaluation sets are disjoint.
LETTER_PAIRS = {
    'train': [('T', 'I'), ('B', '8'), ('S', '5'), ('W', 'V'), ('X', 'K'), ('Z', '2'), ('h', 'n'), ('p', 'q'),
              ('u', 'v'), ('A', 'H'), ('a', 'e'), ('k', 'x')],
    'eval': [('O', 'Q'), ('E', 'F'), ('P', 'R'), ('C', 'G'), ('M', 'N'), ('b', 'd'), ('U', 'J'), ('c', 'o')],
}
WORDS = {
    'train': ['Lamp', 'Rock', 'Tide', 'Moss', 'Gold', 'Fern', 'Dusk', 'Mint', 'Harp', 'Silk', 'Wolf', 'Opal'],
    'eval': ['Song', 'Leaf', 'Rain', 'Star', 'Bird', 'Snow', 'Wind', 'Moon'],
}
TEMPLATES = {'train': st.DEVELOPMENT_TEMPLATES, 'eval': st.HELD_OUT_TEMPLATES}
GROUP_MEMBERS = {'train': ('dot', 'square', 'triangle'), 'eval': ('face',)}
SHAPE_CHANGES = {'train': [('gap', 4), ('spur', 4), ('rotate', 4), ('rotate', 3), ('mirror', 1)],
                 'eval': [('gap', 4), ('spur', 4), ('rotate', 4), ('rotate', 3), ('mirror', 1), ('gap', 3)]}
NOUNS = {'letter': 'letter', 'word': 'word', 'number': 'number', 'shape': 'figure', 'group': 'group of three figures'}
MARK_COLORS = {'red': (230, 0, 0), 'green': (0, 160, 0), 'blue': (0, 0, 230)}

JSON_ONLY = ' Output only the JSON, with no explanation.'
ROWCOL_SUFFIX = (' Give its position. Return only {"row": r, "column": c}, counting rows from top to bottom '
                 'and columns from left to right, starting at 1.')
COORD_NOTE = ' All coordinates use 0-1000 normalization; x increases rightward and y increases downward.'
POINT_SUFFIX = ' Give the center of its bounding box. Return only [{"point_2d": [x, y]}].' + COORD_NOTE
EXTENT_PROMPT = ('Give the tight bounding box of the whole arrangement of {noun}s. Return only '
                 '[{{"bbox_2d": [x_min, y_min, x_max, y_max]}}].' + COORD_NOTE + JSON_ONLY)
FINAL_SUFFIX = ('\nGive your final answer on its own last line as "Final answer: <answer>". '
                "Follow the question's answer format.")
# BabyVision-style question phrasings (paraphrases written for this experiment; the answer format follows BabyVision).
ODD_QUESTIONS = (
    'The image shows a total of {n} {noun}s arranged in {R} rows and {C} columns. One of them is different from the '
    'others. Which row and column is it in? The answer format is (x,y). (For example, the answer for the 2nd row and '
    '3rd column is (2,3)).',
    'There are {n} {noun}s in the image, arranged in {R} rows and {C} columns. One of them is different from the '
    'others. What is the coordinate of this different element? (The X-th row and Y-th column are represented by (X,Y))',
    'The image shows {R} rows and {C} columns, totaling {n} {noun}s. All but one of them are identical. Each '
    'coordinate can be represented as (x,y), for example, the coordinate of the 2nd column in the 1st row is (1,2). '
    'What is the coordinate of the {noun} that is different?',
)
MARK_QUESTIONS = (
    'The image shows a total of {n} {noun}s arranged in {R} rows and {C} columns. One of them is drawn in {color}. '
    'Which row and column is it in? The answer format is (x,y). (For example, the answer for the 2nd row and 3rd '
    'column is (2,3)).',
)


@dataclass
class Replica:
    """One replica grid. target / marked: 0-based (row, col). boxes: per entity (row-major) tight pixel boxes."""
    id: str
    split: str          # 'train' or 'eval' content set
    family: str
    rows: int
    cols: int
    canvas: tuple[int, int]
    cell: tuple[float, float]          # cell width / height in pixels
    target: tuple[int, int]
    noun: str
    content: str                       # short description of common / odd content
    color_mode: str
    image: str = ''
    marked_image: str = ''
    mark_color: str = ''
    boxes: list = field(default_factory=list)
    extent: list = field(default_factory=list)   # tight box of all entities (pixels)

    def tokens_per_cell(self) -> tuple[float, float]:
        return self.cell[0] / 32, self.cell[1] / 32

    def density(self) -> str:
        """Bin of the smaller cell side in merged tokens (32 pixels): '<1', '1-2', '>=2'."""
        m = min(self.tokens_per_cell())
        return '<1' if m < 1 else ('1-2' if m < 2 else '>=2')


# ----------------------------------------------------------------------------- rendering

def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_DIR + name, size)


def _text_tile(text: str, w: int, h: int, font: str, color) -> np.ndarray:
    """Black-on-white (or coloured) text centred in a w x h tile, scaled to fill ~80% of the tighter side."""
    size = max(6, int(min(h * 0.85, w * 0.95 / max(1, 0.62 * len(text)))))
    for _ in range(30):
        f = _font(font, size)
        x0, y0, x1, y1 = f.getbbox(text)
        if (x1 - x0) <= 0.88 * w and (y1 - y0) <= 0.82 * h:
            break
        size -= 1
    img = Image.new('RGB', (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = f.getbbox(text)
    d.text(((w - (x1 - x0)) / 2 - x0, (h - (y1 - y0)) / 2 - y0), text, font=f, fill=color)
    return np.asarray(img)


def _shape_tile(template: str, change: st.Change, w: int, h: int, color) -> np.ndarray:
    side = max(8, int(min(w, h) * 0.9))
    inst = st.render_instance(template, change, 64)
    ink = (inst < 255).any(-1).astype(np.uint8) * 255
    m = np.asarray(Image.fromarray(ink).resize((side, side), Image.Resampling.LANCZOS)) > 90
    out = np.full((h, w, 3), 255, np.uint8)
    oy, ox = (h - side) // 2, (w - side) // 2
    out[oy:oy + side, ox:ox + side][m] = color
    return out


def _member(draw: ImageDraw.ImageDraw, kind: str, cx: float, cy: float, r: float, color, odd: bool, lw: int) -> None:
    """One member of a group cell. odd=True draws the changed variant."""
    if kind == 'face':
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=lw)
        e = r * 0.35
        draw.ellipse((cx - e - lw, cy - e - lw, cx - e + lw, cy - e + lw), fill=color)
        if not odd:   # the odd face has only one eye
            draw.ellipse((cx + e - lw, cy - e - lw, cx + e + lw, cy - e + lw), fill=color)
        draw.arc((cx - r * 0.5, cy - r * 0.2, cx + r * 0.5, cy + r * 0.6), 20, 160, fill=color, width=lw)
    elif kind == 'dot':
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=lw)
        if not odd:   # the odd member lacks the centre dot
            draw.ellipse((cx - r * 0.3, cy - r * 0.3, cx + r * 0.3, cy + r * 0.3), fill=color)
    elif kind == 'square':
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), outline=color, width=lw)
        if odd:       # the odd member has a diagonal
            draw.line((cx - r, cy - r, cx + r, cy + r), fill=color, width=lw)
    elif kind == 'triangle':
        pts = [(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        if odd:       # the odd member points down
            pts = [(cx, cy + r), (cx + r, cy - r), (cx - r, cy - r)]
        draw.polygon(pts, outline=color, width=lw)
    else:
        raise ValueError(kind)


def _group_tile(kind: str, odd_member: int | None, w: int, h: int, color) -> np.ndarray:
    img = Image.new('RGB', (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # Members fill the central 78% of the cell so that neighbouring groups are separated by a visible gap.
    inner = 0.78 * w
    r = min(inner / 6.6, h * 0.42)
    lw = max(1, int(round(r / 6)))
    for j in range(3):
        _member(d, kind, (w - inner) / 2 + inner * (j + 0.5) / 3, h / 2, r, color, odd_member == j, lw)
    return np.asarray(img)


def _dark_color(rng: np.random.Generator) -> tuple[int, int, int]:
    while True:
        c = rng.integers(0, 200, 3)
        if c.max() < 200 and (c.max() - c.min() > 60 or c.max() < 70):
            return tuple(int(v) for v in c)


def sample_layout(rng: np.random.Generator) -> tuple[int, int, int, int, float, float]:
    """(rows, cols, image width, image height, cell w, cell h) within the BabyVision ranges: 7-28 entities per axis,
    cells 19-94 px, image sides 270-740 px, at most 400 entities."""
    while True:
        R, C = int(rng.integers(7, 26)), int(rng.integers(7, 29))
        if R * C > 400:
            continue
        cw, ch = float(rng.uniform(19, 94)), float(rng.uniform(19, 70))
        if not 0.5 <= cw / ch <= 3.2:
            continue
        gw, gh = cw * C, ch * R
        if not (480 <= gw <= 720 and 250 <= gh <= 720):
            continue
        return R, C, int(math.ceil(gw + 24)), int(math.ceil(gh + 24)), cw, ch


def pad32(img: Image.Image) -> Image.Image:
    w, h = img.size
    W, H = -(-max(w, 256) // 32) * 32, -(-max(h, 256) // 32) * 32
    out = Image.new('RGB', (W, H), (255, 255, 255))
    out.paste(img, (0, 0))
    return out


def make_replica(idx: int, split: str, family: str, seed: int) -> tuple[Replica, Image.Image, Image.Image]:
    """Render one replica grid and its marked variant (the target's foreground recoloured)."""
    rng = np.random.default_rng(seed)
    R, C, W, H, cw, ch = sample_layout(rng)
    target = (int(rng.integers(R)), int(rng.integers(C)))
    color_mode = str(rng.choice(['black', 'dark', 'mixed'] if family in ('letter', 'word', 'number') else ['black', 'dark']))
    base_color = (0, 0, 0) if color_mode == 'black' else _dark_color(rng)
    font = str(rng.choice(FONTS))
    if family == 'letter':
        common, odd = LETTER_PAIRS[split][int(rng.integers(len(LETTER_PAIRS[split])))]
        if rng.random() < 0.5:
            common, odd = odd, common
        tile = lambda is_odd, col, w, h: _text_tile(odd if is_odd else common, w, h, font, col)
        content = f'{common}/{odd}'
    elif family == 'word':
        word = WORDS[split][int(rng.integers(len(WORDS[split])))]
        k = int(rng.integers(1, len(word) - 2))
        odd = word[:k] + word[k + 1] + word[k] + word[k + 2:]       # swap two adjacent interior letters
        tile = lambda is_odd, col, w, h: _text_tile(odd if is_odd else word, w, h, font, col)
        content = f'{word}/{odd}'
    elif family == 'number':
        n = int(rng.integers(10, 1000))
        s = str(n)
        k = int(rng.integers(len(s)))
        digits = [d for d in '0123456789' if d != s[k] and not (k == 0 and d == '0')]
        odd = s[:k] + str(rng.choice(digits)) + s[k + 1:]
        tile = lambda is_odd, col, w, h: _text_tile(odd if is_odd else s, w, h, font, col)
        content = f'{s}/{odd}'
    elif family == 'shape':
        while True:
            tpl = str(rng.choice(TEMPLATES[split]))
            kind, lev = SHAPE_CHANGES[split][int(rng.integers(len(SHAPE_CHANGES[split])))]
            if tpl not in st.SYMMETRY_EXCLUSIONS.get(kind, set()):
                break
        change = st.Change(kind, lev, int(rng.integers(3)) if kind in ('gap', 'spur') else 0)
        tile = lambda is_odd, col, w, h: _shape_tile(tpl, change if is_odd else st.NONE, w, h, col)
        content = f'{tpl}/{kind}{lev}'
    elif family == 'group':
        kind = str(rng.choice(GROUP_MEMBERS[split]))
        om = int(rng.integers(3))
        tile = lambda is_odd, col, w, h: _group_tile(kind, om if is_odd else None, w, h, col)
        content = f'{kind}/member{om}'
        if C > 16:   # groups of three need wider cells
            C = 16
    else:
        raise ValueError(family)
    if family == 'group':
        target = (target[0], min(target[1], C - 1))
        cw = min(94.0, max(cw, 54.0))
        W = int(math.ceil(cw * C + 24))
    canvas = np.full((H, W, 3), 255, np.uint8)
    boxes = []
    ox, oy = 12, 12
    for r in range(R):
        for c in range(C):
            x0, y0 = int(round(ox + c * cw)), int(round(oy + r * ch))
            x1, y1 = int(round(ox + (c + 1) * cw)), int(round(oy + (r + 1) * ch))
            col = _dark_color(rng) if color_mode == 'mixed' else base_color
            t = tile((r, c) == target, col, x1 - x0, y1 - y0)
            canvas[y0:y1, x0:x1] = np.minimum(canvas[y0:y1, x0:x1], t)
            ink = np.argwhere((t < 255 - FOREGROUND_THRESHOLD // 2).any(-1))
            if len(ink) == 0:
                raise ValueError('empty tile')
            (ty0, tx0), (ty1, tx1) = ink.min(0), ink.max(0) + 1
            boxes.append([float(x0 + tx0), float(y0 + ty0), float(x0 + tx1), float(y0 + ty1)])
    img = pad32(Image.fromarray(canvas))
    # Marked variant: recolour the target's foreground with the palette colour farthest from the entity colours.
    ref = np.array(base_color)
    cname = max(MARK_COLORS, key=lambda n: np.linalg.norm(np.array(MARK_COLORS[n]) - ref))
    if color_mode == 'mixed':
        cname = 'red'
    a = np.asarray(img).copy()
    x0, y0 = int(round(ox + target[1] * cw)), int(round(oy + target[0] * ch))
    x1, y1 = int(round(ox + (target[1] + 1) * cw)), int(round(oy + (target[0] + 1) * ch))
    cell = a[y0:y1, x0:x1]
    cell[(255 - cell.astype(int)).max(-1) > FOREGROUND_THRESHOLD] = MARK_COLORS[cname]
    marked = Image.fromarray(a)
    b = np.array(boxes)
    extent = [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())]
    rep = Replica(f'{split}_{family}_{idx:04d}', split, family, R, C, img.size, (cw, ch), target, NOUNS[family],
                  content, color_mode, mark_color=cname, boxes=boxes, extent=extent)
    return rep, img, marked


# ----------------------------------------------------------------------------- prompts and answers

def to_units(v: float, size: float) -> int:
    return int(min(1000, max(0, round(1000 * v / size))))


def target_point_units(rep: Replica) -> tuple[int, int]:
    b = rep.boxes[rep.target[0] * rep.cols + rep.target[1]]
    W, H = rep.canvas
    return to_units((b[0] + b[2]) / 2, W), to_units((b[1] + b[3]) / 2, H)


def extent_units(rep: Replica) -> list[int]:
    W, H = rep.canvas
    e = rep.extent
    return [to_units(e[0], W), to_units(e[1], H), to_units(e[2], W), to_units(e[3], H)]


def question(rep: Replica, kind: str, variant: int = 0) -> str:
    """kind: odd_rc / odd_pt / mark_rc / mark_pt (direct) or odd_cot / mark_cot (BabyVision-style reasoning)."""
    noun = rep.noun
    if kind == 'odd_rc':
        return f'One {noun} differs from all the others.' + ROWCOL_SUFFIX + JSON_ONLY
    if kind == 'odd_pt':
        return f'One {noun} differs from all the others.' + POINT_SUFFIX + JSON_ONLY
    if kind == 'mark_rc':
        return f'One {noun} is drawn in {rep.mark_color}.' + ROWCOL_SUFFIX + JSON_ONLY
    if kind == 'mark_pt':
        return f'One {noun} is drawn in {rep.mark_color}.' + POINT_SUFFIX + JSON_ONLY
    fmt = dict(n=rep.rows * rep.cols, noun=noun, R=rep.rows, C=rep.cols, color=rep.mark_color)
    if kind == 'odd_cot':
        return ODD_QUESTIONS[variant % len(ODD_QUESTIONS)].format(**fmt) + FINAL_SUFFIX
    if kind == 'mark_cot':
        return MARK_QUESTIONS[variant % len(MARK_QUESTIONS)].format(**fmt) + FINAL_SUFFIX
    raise ValueError(kind)


def mediated_answer(rep: Replica, point: tuple[int, int] | None = None, extent: list[int] | None = None,
                    marked: bool = False) -> tuple[str, tuple[int, int]]:
    """Coordinate-mediated reasoning answer: write the target point and the grid extent (0-1000 units), divide the
    extent by the stated column / row counts, and convert the target offset into 1-based ordinals with the shown
    numbers. Returns (text, (row, col) implied by the written arithmetic)."""
    x, y = point or target_point_units(rep)
    X0, Y0, X1, Y1 = extent or extent_units(rep)
    cw, rh = (X1 - X0) / rep.cols, (Y1 - Y0) / rep.rows
    u, v = (x - X0) / cw, (y - Y0) / rh
    c = min(rep.cols, max(1, int(math.floor(u)) + 1))
    r = min(rep.rows, max(1, int(math.floor(v)) + 1))
    what = f'the {rep.noun} drawn in {rep.mark_color}' if marked else f'the {rep.noun} that differs from the others'
    text = (f'The grid has {rep.rows} rows and {rep.cols} columns. The center of {what} is at about [{x}, {y}], and the '
            f'whole grid spans [{X0}, {Y0}, {X1}, {Y1}] (0-1000 coordinates).\n'
            f'Each column is ({X1} - {X0}) / {rep.cols} = {cw:.1f} wide, so the column is ({x} - {X0}) / {cw:.1f} = '
            f'{u:.2f}, which is column {c}.\n'
            f'Each row is ({Y1} - {Y0}) / {rep.rows} = {rh:.1f} tall, so the row is ({y} - {Y0}) / {rh:.1f} = '
            f'{v:.2f}, which is row {r}.\n'
            f'Final answer: ({r},{c})')
    return text, (r, c)


def point_answer(rep: Replica, label: str) -> str:
    x, y = target_point_units(rep)
    return '```json\n[\n\t{"point_2d": [%d, %d], "label": "%s"}\n]\n```' % (x, y, label.replace('"', "'"))


def extent_answer(rep: Replica) -> str:
    e = extent_units(rep)
    return '```json\n[\n\t{"bbox_2d": [%d, %d, %d, %d], "label": "%ss"}\n]\n```' % (*e, rep.noun)


def rowcol_json(rep: Replica) -> str:
    return json.dumps({'row': rep.target[0] + 1, 'column': rep.target[1] + 1})


# ----------------------------------------------------------------------------- parsing and error classes

def final_rowcol(text: str) -> tuple[int, int] | None:
    """(row, col) from the 'Final answer' line (last one); else from the last line with two integers."""
    final = text.split('</think>')[-1]
    m = re.findall(r'final answer\s*[:：]\s*\**\s*\(?\s*(\d+)\s*[,，]\s*(\d+)', final, re.I)
    if m:
        return int(m[-1][0]), int(m[-1][1])
    for line in reversed([l for l in final.splitlines() if l.strip()]):
        nums = re.findall(r'\d+', line)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
    return None


def json_rowcol(text: str) -> tuple[int, int] | None:
    mr, mc = re.search(r'"row"\s*:\s*(\d+)', text), re.search(r'"column"\s*:\s*(\d+)', text)
    return (int(mr.group(1)), int(mc.group(1))) if mr and mc else None


CANNOT_SEE = re.compile(r"(can(?:no|')t see|cannot (?:clearly )?(?:see|view|identify|determine|distinguish|make out)|"
                        r"unable to (?:see|view|identify|determine|distinguish)|without (?:the |seeing the |access to the )?"
                        r"image|(?:image|picture) (?:is|was) not (?:provided|visible|available|shown)|not clear enough|"
                        r"too (?:small|blurry)|i (?:don't|do not) have (?:access|the image))", re.I)


def format_class(text: str, finish_reason: str, rc) -> str | None:
    """'truncated' / 'cannot_see' / 'unparsed' for format and refusal failures, else None."""
    if finish_reason == 'length':
        return 'truncated'
    if CANNOT_SEE.search(text):
        return 'cannot_see'
    if rc is None:
        return 'unparsed'
    return None


def cell_of_point(pt, rep_rows: int, rep_cols: int, boxes: list, canvas) -> tuple[int, int] | None:
    """Grid cell (0-based) whose entity box centre is nearest to a 0-1000 point answer."""
    if pt is None:
        return None
    W, H = canvas
    px, py = pt[0] * W / 1000, pt[1] * H / 1000
    b = np.asarray(boxes, float)
    c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    k = int(np.argmin((c[:, 0] - px) ** 2 + (c[:, 1] - py) ** 2))
    return k // rep_cols, k % rep_cols


ERROR_CLASSES = ('correct', 'format', 'ordinal', 'detection')


def diagnose(odd_rc, odd_fmt: str | None, mark_rc, truth: tuple[int, int], odd_cell, near: int = 1) -> str:
    """Error class of one find-the-difference answer (1-based truth), from paired answers on the same grid.

    * format: the find-the-difference answer is truncated, claims the image cannot be seen, or has no row/column;
    * ordinal: the model located the target but wrote a wrong row/column; evidence (any one): the answer is within
      `near` of the truth on both axes, or the answer equals the model's own wrong answer for the marked target, or
      the odd-entity point answer lies in the target cell or a neighbouring cell (Chebyshev distance <= near);
    * detection: otherwise (the answer is far from the target and no evidence that the target was found).
    """
    if odd_fmt is not None:
        return 'format'
    if tuple(odd_rc) == tuple(truth):
        return 'correct'
    d = max(abs(odd_rc[0] - truth[0]), abs(odd_rc[1] - truth[1]))
    if d <= near:
        return 'ordinal'
    if mark_rc is not None and tuple(mark_rc) != tuple(truth) and tuple(mark_rc) == tuple(odd_rc):
        return 'ordinal'
    if odd_cell is not None and max(abs(odd_cell[0] + 1 - truth[0]), abs(odd_cell[1] + 1 - truth[1])) <= near:
        return 'ordinal'
    return 'detection'


# ----------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float('nan'), float('nan')]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b (base right, other wrong) vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ----------------------------------------------------------------------------- adapter conditions

L1_4 = (1, 2, 3, 4)
GATE_LAYERS = (6, 7, 9, 10)        # 7.3C pre-registered counting-gate layers
VIT_SCOPE = 'vit_mlp:25-27'         # 7.3 block-window scan: best held-out-family transfer
# Each condition: decoder part ('mlp': layers whose MLP gets LoRA; 'gate': DeltaNet layers whose in_proj_a/b get LoRA;
# 'all_linear': every decoder linear layer), its token positions ('image' or None = all), ViT scope (or None),
# learning rates per part, and the seed.
CONDITIONS: dict[str, dict] = {
    'main': {'mlp': L1_4, 'gate': GATE_LAYERS, 'vit': VIT_SCOPE, 'positions': 'image', 'seed': 1},
    'main_s2': {'mlp': L1_4, 'gate': GATE_LAYERS, 'vit': VIT_SCOPE, 'positions': 'image', 'seed': 2},
    'mlp_only': {'mlp': L1_4, 'gate': (), 'vit': None, 'positions': 'image', 'seed': 1},
    'gate_only': {'mlp': (), 'gate': GATE_LAYERS, 'vit': None, 'positions': 'image', 'seed': 1},
    'vit_only': {'mlp': (), 'gate': (), 'vit': VIT_SCOPE, 'positions': None, 'seed': 1},
    'full_lora': {'all_linear': True, 'mlp': (), 'gate': (), 'vit': None, 'positions': None, 'seed': 1},
    # Revision after the main result (re-predicted before training, results doc section 5.2): the same located modules
    # with the decoder update applied at every token position, so that text positions can learn the written
    # coordinate-mediated procedure (an image-token-only update leaves every generated token on the base weights).
    'main_alltok': {'mlp': L1_4, 'gate': GATE_LAYERS, 'vit': VIT_SCOPE, 'positions': None, 'seed': 1},
    'main_alltok_s2': {'mlp': L1_4, 'gate': GATE_LAYERS, 'vit': VIT_SCOPE, 'positions': None, 'seed': 2},
    'mlp_alltok': {'mlp': L1_4, 'gate': (), 'vit': None, 'positions': None, 'seed': 1},
    # Second revision (section 3.6): the all-token adapter learned the opening sentence of the procedure but skipped
    # the coordinate steps after one epoch; three epochs over the same 2000 samples.
    'main_alltok_e3': {'mlp': L1_4, 'gate': GATE_LAYERS, 'vit': VIT_SCOPE, 'positions': None, 'seed': 1, 'epochs': 3},
    'main_alltok_e3_s2': {'mlp': L1_4, 'gate': GATE_LAYERS, 'vit': VIT_SCOPE, 'positions': None, 'seed': 2, 'epochs': 3},
    'mlp_alltok_e3': {'mlp': L1_4, 'gate': (), 'vit': None, 'positions': None, 'seed': 1, 'epochs': 3},
    'main_e3': {'mlp': L1_4, 'gate': GATE_LAYERS, 'vit': VIT_SCOPE, 'positions': 'image', 'seed': 1, 'epochs': 3},
    # Data ablation of the main components: no coordinate-mediated reasoning samples (direct row/column only).
    'main_nocot': {'mlp': L1_4, 'gate': GATE_LAYERS, 'vit': VIT_SCOPE, 'positions': 'image', 'seed': 1,
                   'train_file': 'train_nocot.jsonl'},
}
LR = {'mlp': 1e-4, 'gate': 2e-4, 'vit': 2e-4, 'all_linear': 1e-4}


def dumps(obj) -> str:
    return json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


def replica_dict(rep: Replica) -> dict:
    return asdict(rep)


def arithmetic_consistency(text: str, rows: int, cols: int) -> dict | None:
    """For a coordinate-mediated reasoning answer: the (row, col) implied by its own written target point and grid
    extent (section 3.1 formula), and whether the final answer equals it. None if the numbers are not written."""
    mp = re.search(r'at about \[(\d+),\s*(\d+)\]', text)
    me = re.search(r'spans \[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text)
    if not mp or not me:
        return None
    x, y = int(mp.group(1)), int(mp.group(2))
    X0, Y0, X1, Y1 = (int(v) for v in me.groups())
    if X1 <= X0 or Y1 <= Y0:
        return None
    c = min(cols, max(1, math.floor((x - X0) / ((X1 - X0) / cols)) + 1))
    r = min(rows, max(1, math.floor((y - Y0) / ((Y1 - Y0) / rows)) + 1))
    fin = final_rowcol(text)
    return {'implied': [r, c], 'final': None if fin is None else list(fin), 'consistent': fin is not None and list(fin) == [r, c],
            'point': [x, y], 'extent': [X0, Y0, X1, Y1]}
