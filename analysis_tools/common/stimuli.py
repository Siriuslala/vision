"""Shared synthetic stimulus generator (docs/analysis/stimuli.md).

Every image is fully determined by a ``StimulusSpec``.  Instances are drawn from
vector primitives in a 64x64 template frame and rendered directly at the cell
scale, so rotation, scaling, and mirroring never resample rasters.  Gaps are
produced by erasing only the pixels that belong to one stroke inside an interval
along that stroke, which guarantees that the altered instance differs from the
template only at the intended stroke.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import io
import math

import numpy as np
from PIL import Image, ImageDraw

GENERATOR_VERSION = 'stimuli-v1'
TOKEN_PIXELS = 32          # one merged visual token covers 32x32 input pixels
MIN_CANVAS = 256           # processor shortest_edge=65536 pixels => 256x256 avoids upscaling
BASE = 64                  # template frame size
WHITE = 255

DEVELOPMENT_TEMPLATES = ('person', 'ring', 'square', 'arrow')
HELD_OUT_TEMPLATES = ('star', 'house')
TEMPLATES = DEVELOPMENT_TEMPLATES + HELD_OUT_TEMPLATES

# Magnitude ladders; index 1..4 maps to level 1..4.  Level 0 means "unchanged".
LEVELS: dict[str, tuple] = {
    'gap': (2, 4, 8, 16),            # erased length along the stroke, base pixels
    'spur': (2, 4, 8, 16),           # added stroke length, base pixels
    'color': (5, 10, 20, 'red'),     # CIELAB Delta E76 from black, or pure red
    'rotate': (5, 15, 45, 90),       # degrees, clockwise in image coordinates
    'scale': (0.9, 0.8, 0.7, 0.5),   # isotropic scale about the template centre
    'mirror': (True,),               # horizontal flip
}
# Templates excluded for a change type because the change leaves them identical.
SYMMETRY_EXCLUSIONS = {
    'rotate': {'ring', 'square'},
    'mirror': {'ring', 'square', 'star', 'person'},
}


@dataclass(frozen=True)
class Line:
    p: tuple[float, float]
    q: tuple[float, float]


@dataclass(frozen=True)
class Circle:
    c: tuple[float, float]
    r: float


def _poly(points: list[tuple[float, float]], closed: bool = True) -> list[Line]:
    pairs = list(zip(points, points[1:] + (points[:1] if closed else [])))
    return [Line(a, b) for a, b in pairs]


def _star_points() -> list[tuple[float, float]]:
    pts = []
    for k in range(10):
        radius = 22 if k % 2 == 0 else 9
        angle = -math.pi / 2 + k * math.pi / 5
        pts.append((32 + radius * math.cos(angle), 33 + radius * math.sin(angle)))
    return pts


# Template definitions: primitives, stroke width (base pixels), and three
# (primitive index, position along the primitive) sites used for gaps and spurs.
# Person, ring, and square reuse the historical geometry of visual_detail.py and
# distance_register.py.
TEMPLATE_DEFS: dict[str, dict] = {
    'person': {
        'width': 2,
        'prims': [Circle((32, 16), 8), Line((32, 24), (32, 44)), Line((16, 32), (48, 32)),
                  Line((32, 44), (20, 57)), Line((32, 44), (44, 57))],
        'sites': [(2, 0.83), (3, 0.6), (0, 0.5)],
    },
    'ring': {'width': 3, 'prims': [Circle((31, 31), 19)], 'sites': [(0, 0.0), (0, 1 / 3), (0, 2 / 3)]},
    'square': {'width': 3, 'prims': _poly([(12, 12), (50, 12), (50, 50), (12, 50)]),
               'sites': [(0, 0.5), (1, 0.3), (2, 0.7)]},
    'arrow': {'width': 2, 'prims': [Line((12, 32), (52, 32)), Line((52, 32), (42, 22)), Line((52, 32), (42, 42))],
              'sites': [(0, 0.35), (0, 0.65), (1, 0.6)]},
    'star': {'width': 2, 'prims': _poly(_star_points()), 'sites': [(0, 0.5), (3, 0.5), (6, 0.5)]},
    'house': {'width': 2,
              'prims': [Line((14, 30), (14, 54)), Line((14, 54), (50, 54)), Line((50, 54), (50, 30)),
                        Line((10, 30), (32, 10)), Line((32, 10), (54, 30)), Line((10, 30), (54, 30)),
                        Line((20, 54), (20, 42)), Line((20, 42), (28, 42)), Line((28, 42), (28, 54)),
                        *_poly([(36, 36), (44, 36), (44, 44), (36, 44)])],
              'sites': [(4, 0.5), (3, 0.5), (0, 0.2)]},
}


@dataclass(frozen=True)
class Change:
    """One attribute change applied to an instance; ``level`` 0 means the template."""
    kind: str = 'none'
    level: int = 0
    site: int = 0

    def __post_init__(self):
        if self.kind == 'none':
            if self.level != 0:
                raise ValueError('kind "none" requires level 0')
            return
        if self.kind not in LEVELS:
            raise ValueError(f'Unknown change kind {self.kind!r}')
        if not 1 <= self.level <= len(LEVELS[self.kind]):
            raise ValueError(f'Level {self.level} outside 1..{len(LEVELS[self.kind])} for {self.kind}')
        if self.kind in ('gap', 'spur') and self.site not in (0, 1, 2):
            raise ValueError('gap/spur site must be 0, 1, or 2')

    @property
    def magnitude(self):
        return None if self.kind == 'none' else LEVELS[self.kind][self.level - 1]


NONE = Change()


@dataclass(frozen=True)
class StimulusSpec:
    """Complete description of one stimulus image.

    target: zero-based (row, col) of the minority instance, or None for a normal image.
    swap: if True, the target keeps the template and every other instance carries ``change``.
    marker: zero-based (row, col) whose cell border is drawn in blue, or None.
    """
    template: str
    rows: int
    cols: int
    change: Change = NONE
    target: tuple[int, int] | None = None
    swap: bool = False
    cell: int = 64
    phase: int = 0
    jitter: int = 0
    marker: tuple[int, int] | None = None
    seed: int = 0
    canvas_min: int = MIN_CANVAS

    def __post_init__(self):
        if self.template not in TEMPLATE_DEFS:
            raise ValueError(f'Unknown template {self.template!r}')
        if self.rows < 1 or self.cols < 1:
            raise ValueError('rows and cols must be positive')
        if self.cell % TOKEN_PIXELS:
            raise ValueError('cell must be a multiple of 32 pixels')
        if self.phase not in (0, 8, 16):
            raise ValueError('phase must be 0, 8, or 16 pixels')
        if not 0 <= self.jitter <= self.cell // 8:
            raise ValueError('jitter must lie in [0, cell/8]')
        if self.phase and self.jitter:
            raise ValueError('phase offset and jitter are separate conditions')
        if self.marker is not None and self.jitter:
            raise ValueError('markers require jitter=0 so the border keeps its clearance')
        for name, rc in (('target', self.target), ('marker', self.marker)):
            if rc is not None and not (0 <= rc[0] < self.rows and 0 <= rc[1] < self.cols):
                raise ValueError(f'{name} {rc} outside a {self.rows}x{self.cols} grid')
        if self.change.kind != 'none' and self.target is None:
            raise ValueError('A change requires a target instance')
        if self.swap and self.change.kind == 'none':
            raise ValueError('swap requires a change')
        if self.template in SYMMETRY_EXCLUSIONS.get(self.change.kind, set()):
            raise ValueError(f'{self.change.kind} leaves template {self.template!r} unchanged')


def lab_to_srgb(lab: tuple[float, float, float]) -> tuple[int, int, int]:
    """CIELAB (D65) to 8-bit sRGB, raising if the colour is outside the gamut."""
    L, a, b = lab
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    def finv(t):
        return t ** 3 if t ** 3 > 0.008856 else (t - 16 / 116) / 7.787
    X, Y, Z = 0.95047 * finv(fx), 1.0 * finv(fy), 1.08883 * finv(fz)
    rgb_lin = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]]) @ np.array([X, Y, Z])
    if (rgb_lin < -1e-6).any() or (rgb_lin > 1 + 1e-6).any():
        raise ValueError(f'Lab colour {lab} is outside the sRGB gamut')
    rgb_lin = rgb_lin.clip(0, 1)
    rgb = np.where(rgb_lin <= 0.0031308, 12.92 * rgb_lin, 1.055 * rgb_lin ** (1 / 2.4) - 0.055)
    return tuple(int(round(v * 255)) for v in rgb)


def stroke_color(change: Change) -> tuple[int, int, int]:
    if change.kind != 'color':
        return (0, 0, 0)
    magnitude = change.magnitude
    if magnitude == 'red':
        return (255, 0, 0)
    # Fixed reddish hue (40 degrees); Delta E76 from black equals ``magnitude``.
    hue = math.radians(40)
    lab = (0.5 * magnitude, 0.866 * magnitude * math.cos(hue), 0.866 * magnitude * math.sin(hue))
    return lab_to_srgb(lab)


def _transform(change: Change):
    """Return a point map in template coordinates for geometric changes."""
    cx = cy = BASE / 2
    if change.kind == 'rotate':
        t = math.radians(change.magnitude)
        c, s = math.cos(t), math.sin(t)
        return lambda p: (cx + c * (p[0] - cx) - s * (p[1] - cy), cy + s * (p[0] - cx) + c * (p[1] - cy)), 1.0
    if change.kind == 'scale':
        k = change.magnitude
        return lambda p: (cx + k * (p[0] - cx), cy + k * (p[1] - cy)), k
    if change.kind == 'mirror':
        return lambda p: (BASE - p[0], p[1]), 1.0
    return lambda p: p, 1.0


def _draw_prim(draw: ImageDraw.ImageDraw, prim, scale: float, width: int, fill) -> None:
    if isinstance(prim, Line):
        draw.line((prim.p[0] * scale, prim.p[1] * scale, prim.q[0] * scale, prim.q[1] * scale), fill=fill, width=width)
    else:
        r = prim.r * scale
        cx, cy = prim.c[0] * scale, prim.c[1] * scale
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=fill, width=width)


def _prim_mask(prim, size: int, scale: float, width: int) -> np.ndarray:
    im = Image.new('L', (size, size), 0)
    _draw_prim(ImageDraw.Draw(im), prim, scale, width, 255)
    return np.asarray(im) > 0


def _site_point(prim, t: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Point on a primitive and the unit outward normal at that point (template coordinates)."""
    if isinstance(prim, Line):
        p = (prim.p[0] + t * (prim.q[0] - prim.p[0]), prim.p[1] + t * (prim.q[1] - prim.p[1]))
        dx, dy = prim.q[0] - prim.p[0], prim.q[1] - prim.p[1]
        n = math.hypot(dx, dy)
        normal = (-dy / n, dx / n)
        # Orient away from the template centre so spurs point outwards.
        if (p[0] - BASE / 2) * normal[0] + (p[1] - BASE / 2) * normal[1] < 0:
            normal = (-normal[0], -normal[1])
        return p, normal
    angle = 2 * math.pi * t
    normal = (math.cos(angle), math.sin(angle))
    return (prim.c[0] + prim.r * normal[0], prim.c[1] + prim.r * normal[1]), normal


def _gap_region(prim, t: float, length: float, size: int, scale: float) -> np.ndarray:
    """Pixels whose projection onto the stroke falls inside the gap interval."""
    ys, xs = np.mgrid[0:size, 0:size].astype(float) + 0.5
    xs, ys = xs / scale, ys / scale
    if isinstance(prim, Line):
        dx, dy = prim.q[0] - prim.p[0], prim.q[1] - prim.p[1]
        total = math.hypot(dx, dy)
        along = ((xs - prim.p[0]) * dx + (ys - prim.p[1]) * dy) / total
        centre = t * total
        return np.abs(along - centre) <= length / 2
    angle = np.arctan2(ys - prim.c[1], xs - prim.c[0])
    delta = np.angle(np.exp(1j * (angle - 2 * math.pi * t)))
    return np.abs(delta) * prim.r <= length / 2


def _spur_direction(p, normal, width) -> tuple[float, float]:
    """Outward normal unless the longest spur would leave the safe area; then point inwards.

    The direction depends only on the site, so every magnitude level grows the same spur.
    """
    longest = max(LEVELS['spur']) + width / 2
    end = (p[0] + normal[0] * longest, p[1] + normal[1] * longest)
    if all(4 <= v <= BASE - 4 for v in end):
        return normal
    return (-normal[0], -normal[1])


def render_instance(template: str, change: Change = NONE, cell: int = 64) -> np.ndarray:
    """Render one instance as a (cell, cell, 3) uint8 array on a white background."""
    spec = TEMPLATE_DEFS[template]
    scale = cell / BASE
    width = max(1, round(spec['width'] * scale))
    point_map, radius_scale = _transform(change)
    prims = []
    for prim in spec['prims']:
        if isinstance(prim, Line):
            prims.append(Line(point_map(prim.p), point_map(prim.q)))
        else:
            prims.append(Circle(point_map(prim.c), prim.r * radius_scale))
    color = stroke_color(change)
    strokes = np.zeros((cell, cell), bool)
    masks = [_prim_mask(p, cell, scale, width) for p in prims]
    for m in masks:
        strokes |= m
    if change.kind == 'gap':
        index, t = spec['sites'][change.site]
        erase = masks[index] & _gap_region(prims[index], t, change.magnitude, cell, scale)
        others = np.zeros_like(strokes)
        for j, m in enumerate(masks):
            if j != index:
                others |= m
        strokes &= ~(erase & ~others)   # never erase pixels shared with another stroke
    if change.kind == 'spur':
        index, t = spec['sites'][change.site]
        p, normal = _site_point(prims[index], t)
        normal = _spur_direction(p, normal, spec['width'])
        start = (p[0] + normal[0] * spec['width'] / 2, p[1] + normal[1] * spec['width'] / 2)
        end = (start[0] + normal[0] * change.magnitude, start[1] + normal[1] * change.magnitude)
        strokes |= _prim_mask(Line(start, end), cell, scale, width)
    out = np.full((cell, cell, 3), WHITE, np.uint8)
    out[strokes] = color
    return out


def canvas_layout(spec: StimulusSpec) -> tuple[int, int, int, int]:
    """Canvas width/height and the grid origin; both stay aligned to the 32-pixel token grid."""
    margin = TOKEN_PIXELS if (spec.phase or spec.jitter) else 0
    shift = spec.phase + (TOKEN_PIXELS // 2 if spec.jitter else 0)
    def axis(n):
        content = n * spec.cell
        size = math.ceil(max(spec.canvas_min, content + margin) / TOKEN_PIXELS) * TOKEN_PIXELS
        origin = ((size - content - margin) // 2 // TOKEN_PIXELS) * TOKEN_PIXELS
        return size, origin + shift
    width, ox = axis(spec.cols)
    height, oy = axis(spec.rows)
    return width, height, ox, oy


def _instance_change(spec: StimulusSpec, rc: tuple[int, int]) -> Change:
    is_target = spec.target == rc
    return spec.change if is_target != spec.swap else NONE


def generate(spec: StimulusSpec) -> tuple[Image.Image, dict]:
    """Render a stimulus and return (image, metadata) with full ground truth."""
    width, height, ox, oy = canvas_layout(spec)
    canvas = np.full((height, width, 3), WHITE, np.uint8)
    rng = np.random.default_rng(spec.seed)
    cache: dict[Change, np.ndarray] = {}
    instances = []
    for r in range(spec.rows):
        for c in range(spec.cols):
            change = _instance_change(spec, (r, c))
            if change not in cache:
                cache[change] = render_instance(spec.template, change, spec.cell)
            tile = cache[change]
            jx, jy = (rng.integers(-spec.jitter, spec.jitter + 1, 2).tolist() if spec.jitter else (0, 0))
            x0, y0 = ox + c * spec.cell + jx, oy + r * spec.cell + jy
            region = canvas[y0:y0 + spec.cell, x0:x0 + spec.cell]
            np.minimum(region, tile, out=region)
            dark = np.argwhere((tile < WHITE).any(-1))
            (ty0, tx0), (ty1, tx1) = dark.min(0), dark.max(0) + 1   # half-open tight box
            box = [float(x0 + tx0), float(y0 + ty0), float(x0 + tx1), float(y0 + ty1)]
            centre = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            instances.append({
                'row': r, 'col': c, 'index': r * spec.cols + c,
                'cell_box': [ox + c * spec.cell, oy + r * spec.cell, ox + (c + 1) * spec.cell, oy + (r + 1) * spec.cell],
                'tight_box': box, 'center': centre,
                'center_norm': [1000 * centre[0] / width, 1000 * centre[1] / height],
                'change': asdict(change), 'is_target': spec.target == (r, c),
            })
    image = Image.fromarray(canvas)
    if spec.marker is not None:
        r, c = spec.marker
        x0, y0 = ox + c * spec.cell, oy + r * spec.cell
        ImageDraw.Draw(image).rectangle((x0 + 1, y0 + 1, x0 + spec.cell - 2, y0 + spec.cell - 2),
                                        outline=(0, 0, 255), width=2)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    metadata = {
        'generator': GENERATOR_VERSION,
        'spec': {**asdict(spec), 'change': asdict(spec.change)},
        'magnitude': spec.change.magnitude,
        'canvas': [width, height],
        'patch_grid': [height // 16, width // 16],
        'merged_grid': [height // TOKEN_PIXELS, width // TOKEN_PIXELS],
        'grid_origin': [ox, oy],
        'instances': instances,
        'sha256': hashlib.sha256(buffer.getvalue()).hexdigest(),
    }
    return image, metadata


def changed_pixels(template: str, change: Change, cell: int = 64) -> int:
    """Number of pixels that differ between the template and the changed instance."""
    a = render_instance(template, NONE, cell)
    b = render_instance(template, change, cell)
    return int((a != b).any(-1).sum())


def valid_changes(template: str) -> list[Change]:
    """All (kind, level, site) combinations that alter this template."""
    out = []
    for kind, levels in LEVELS.items():
        if template in SYMMETRY_EXCLUSIONS.get(kind, set()):
            continue
        sites = (0, 1, 2) if kind in ('gap', 'spur') else (0,)
        for level in range(1, len(levels) + 1):
            out.extend(Change(kind, level, s) for s in sites)
    return out


def uniform_targets(rows: int, cols: int, count: int, seed: int) -> list[tuple[int, int]]:
    """Uniformly sampled target cells (with replacement across images, fixed seed)."""
    rng = np.random.default_rng(seed)
    flat = rng.integers(0, rows * cols, count)
    return [(int(i) // cols, int(i) % cols) for i in flat]
