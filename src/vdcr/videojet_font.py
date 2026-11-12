"""Videojet-like 16-high bitmap font helpers.

The glyphs are generated from simple stroke segments on a 16-row grid.  This is
closer to the Videojet controller preview than the older stretched 5x7 font:
thin strokes, square terminal bars, diagonal V/X/Y limbs, and compact rounded
D/O/J shapes.
"""
from __future__ import annotations

from collections.abc import Callable

FONT_VERSION = "videojet-like-16-high-60dpi-uniform-thick-v4"
FONT_SOURCE = "handmade-fallback"
FONT_SOURCE_PATH = ""
HEIGHT = 16
TOP = 1
MID = 8
BOTTOM = 15

Canvas = set[tuple[int, int]]
GlyphBuilder = Callable[[Canvas], int]


def _plot(canvas: Canvas, x: int, y: int) -> None:
    if 0 <= y < HEIGHT and x >= 0:
        canvas.add((int(x), int(y)))


def _thick_point(canvas: Canvas, x: int, y: int, *, thickness: int = 2) -> None:
    for xx in range(x, x + max(1, int(thickness))):
        for yy in range(y, y + max(1, int(thickness))):
            _plot(canvas, xx, yy)


def _line(canvas: Canvas, x0: int, y0: int, x1: int, y1: int, *, thickness: int = 2) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        _thick_point(canvas, x, y, thickness=thickness)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _h(canvas: Canvas, x0: int, x1: int, y: int, *, thickness: int = 2) -> None:
    for yy in range(y, y + max(1, int(thickness))):
        for x in range(min(x0, x1), max(x0, x1) + 1):
            _plot(canvas, x, yy)


def _v(canvas: Canvas, x: int, y0: int = TOP, y1: int = BOTTOM, *, thickness: int = 2) -> None:
    for xx in range(x, x + max(1, int(thickness))):
        for y in range(min(y0, y1), max(y0, y1) + 1):
            _plot(canvas, xx, y)


def _arc_right(canvas: Canvas, x_left: int, x_right: int, y_top: int, y_bottom: int) -> None:
    _line(canvas, x_right - 2, y_top, x_right, y_top + 2)
    _v(canvas, x_right, y_top + 3, y_bottom - 3)
    _line(canvas, x_right, y_bottom - 2, x_right - 2, y_bottom)


def _arc_left(canvas: Canvas, x_left: int, x_right: int, y_top: int, y_bottom: int) -> None:
    _line(canvas, x_left + 2, y_top, x_left, y_top + 2)
    _v(canvas, x_left, y_top + 3, y_bottom - 3)
    _line(canvas, x_left, y_bottom - 2, x_left + 2, y_bottom)


def _unknown(canvas: Canvas) -> int:
    _h(canvas, 1, 5, TOP)
    _line(canvas, 5, TOP + 1, 3, 6)
    _v(canvas, 3, 7, 10)
    _plot(canvas, 3, BOTTOM)
    return 7


def _space(canvas: Canvas) -> int:
    return 4


def _glyph_a(c: Canvas) -> int:
    # Videojet-style 16-high A: square cap, double vertical legs, double crossbar.
    _h(c, 1, 6, TOP)
    _v(c, 0, TOP + 2, BOTTOM)
    _v(c, 6, TOP + 2, BOTTOM)
    _h(c, 0, 7, 8)
    return 8


def _glyph_b(c: Canvas) -> int:
    _v(c, 0)
    _h(c, 0, 4, TOP)
    _h(c, 0, 5, MID)
    _h(c, 0, 4, BOTTOM)
    _arc_right(c, 0, 6, TOP, MID)
    _arc_right(c, 0, 6, MID, BOTTOM)
    return 7


def _glyph_c(c: Canvas) -> int:
    _h(c, 2, 6, TOP)
    _h(c, 2, 6, BOTTOM)
    _arc_left(c, 0, 6, TOP, BOTTOM)
    return 7


def _glyph_d(c: Canvas) -> int:
    _v(c, 0)
    _h(c, 0, 4, TOP)
    _h(c, 0, 4, BOTTOM)
    _arc_right(c, 0, 6, TOP, BOTTOM)
    return 7


def _glyph_e(c: Canvas) -> int:
    _v(c, 0)
    _h(c, 0, 6, TOP)
    _h(c, 0, 5, MID)
    _h(c, 0, 6, BOTTOM)
    return 7


def _glyph_f(c: Canvas) -> int:
    _v(c, 0)
    _h(c, 0, 6, TOP)
    _h(c, 0, 5, MID)
    return 7


def _glyph_g(c: Canvas) -> int:
    _h(c, 2, 6, TOP)
    _h(c, 2, 6, BOTTOM)
    _arc_left(c, 0, 6, TOP, BOTTOM)
    _v(c, 6, 9, BOTTOM)
    _h(c, 3, 6, 9)
    return 7


def _glyph_h(c: Canvas) -> int:
    _v(c, 0)
    _v(c, 6)
    _h(c, 0, 6, MID)
    return 7


def _glyph_i(c: Canvas) -> int:
    _h(c, 0, 5, TOP)
    _v(c, 2)
    _h(c, 0, 5, BOTTOM)
    return 6


def _glyph_j(c: Canvas) -> int:
    _h(c, 1, 6, TOP)
    _v(c, 5, TOP, BOTTOM - 3)
    _line(c, 5, BOTTOM - 2, 3, BOTTOM)
    _line(c, 3, BOTTOM, 0, BOTTOM - 4)
    return 7


def _glyph_k(c: Canvas) -> int:
    _v(c, 0)
    _line(c, 6, TOP, 0, MID)
    _line(c, 0, MID, 6, BOTTOM)
    return 7


def _glyph_l(c: Canvas) -> int:
    _v(c, 0)
    _h(c, 0, 6, BOTTOM)
    return 7


def _glyph_m(c: Canvas) -> int:
    _v(c, 0)
    _v(c, 8)
    _line(c, 0, TOP, 4, 7)
    _line(c, 8, TOP, 4, 7)
    return 9


def _glyph_n(c: Canvas) -> int:
    _v(c, 0)
    _v(c, 7)
    _line(c, 0, TOP, 7, BOTTOM)
    return 8


def _glyph_o(c: Canvas) -> int:
    _h(c, 2, 5, TOP)
    _h(c, 2, 5, BOTTOM)
    _arc_left(c, 0, 7, TOP, BOTTOM)
    _arc_right(c, 0, 7, TOP, BOTTOM)
    return 8


def _glyph_p(c: Canvas) -> int:
    _v(c, 0)
    _h(c, 0, 4, TOP)
    _h(c, 0, 5, MID)
    _arc_right(c, 0, 6, TOP, MID)
    return 7


def _glyph_q(c: Canvas) -> int:
    _glyph_o(c)
    _line(c, 4, 11, 7, BOTTOM)
    return 8


def _glyph_r(c: Canvas) -> int:
    _glyph_p(c)
    _line(c, 2, MID, 6, BOTTOM)
    return 7


def _glyph_s(c: Canvas) -> int:
    _h(c, 1, 6, TOP)
    _line(c, 1, TOP, 0, 3)
    _v(c, 0, 3, 6)
    _h(c, 1, 5, MID)
    _v(c, 6, 9, 13)
    _line(c, 6, 13, 5, BOTTOM)
    _h(c, 0, 5, BOTTOM)
    return 7


def _glyph_t(c: Canvas) -> int:
    _h(c, 0, 8, TOP)
    _v(c, 4)
    return 9


def _glyph_u(c: Canvas) -> int:
    _v(c, 0, TOP, BOTTOM - 3)
    _v(c, 6, TOP, BOTTOM - 3)
    _line(c, 0, BOTTOM - 2, 2, BOTTOM)
    _line(c, 6, BOTTOM - 2, 4, BOTTOM)
    _h(c, 2, 4, BOTTOM)
    return 7


def _glyph_v(c: Canvas) -> int:
    _line(c, 0, TOP, 3, BOTTOM)
    _line(c, 6, TOP, 3, BOTTOM)
    return 7


def _glyph_w(c: Canvas) -> int:
    _line(c, 0, TOP, 2, BOTTOM)
    _line(c, 4, TOP, 2, BOTTOM)
    _line(c, 4, TOP, 6, BOTTOM)
    _line(c, 8, TOP, 6, BOTTOM)
    return 9


def _glyph_x(c: Canvas) -> int:
    _line(c, 0, TOP, 6, BOTTOM)
    _line(c, 6, TOP, 0, BOTTOM)
    return 7


def _glyph_y(c: Canvas) -> int:
    _line(c, 0, TOP, 3, 8)
    _line(c, 6, TOP, 3, 8)
    _v(c, 3, 8, BOTTOM)
    return 7


def _glyph_z(c: Canvas) -> int:
    _h(c, 0, 6, TOP)
    _line(c, 6, TOP, 0, BOTTOM)
    _h(c, 0, 6, BOTTOM)
    return 7


def _digit(c: Canvas, ch: str) -> int:
    # Compact seven-segment style, matching the same stroke vocabulary.
    seg = {
        '0': 'abcfed', '1': 'bc', '2': 'abged', '3': 'abgcd', '4': 'fgbc',
        '5': 'afgcd', '6': 'afgecd', '7': 'abc', '8': 'abcdefg', '9': 'abfgcd'
    }[ch]
    if 'a' in seg: _h(c, 1, 5, TOP)
    if 'b' in seg: _v(c, 6, TOP + 1, MID - 1)
    if 'c' in seg: _v(c, 6, MID + 1, BOTTOM - 1)
    if 'd' in seg: _h(c, 1, 5, BOTTOM)
    if 'e' in seg: _v(c, 0, MID + 1, BOTTOM - 1)
    if 'f' in seg: _v(c, 0, TOP + 1, MID - 1)
    if 'g' in seg: _h(c, 1, 5, MID)
    return 7


_BUILDERS: dict[str, GlyphBuilder] = {
    'A': _glyph_a, 'B': _glyph_b, 'C': _glyph_c, 'D': _glyph_d, 'E': _glyph_e,
    'F': _glyph_f, 'G': _glyph_g, 'H': _glyph_h, 'I': _glyph_i, 'J': _glyph_j,
    'K': _glyph_k, 'L': _glyph_l, 'M': _glyph_m, 'N': _glyph_n, 'O': _glyph_o,
    'P': _glyph_p, 'Q': _glyph_q, 'R': _glyph_r, 'S': _glyph_s, 'T': _glyph_t,
    'U': _glyph_u, 'V': _glyph_v, 'W': _glyph_w, 'X': _glyph_x, 'Y': _glyph_y,
    'Z': _glyph_z, '?': _unknown, ' ': _space,
}
for _d in '0123456789':
    _BUILDERS[_d] = (lambda ch: (lambda c: _digit(c, ch)))(_d)


def glyph_points(ch: str) -> tuple[set[tuple[int, int]], int]:
    canvas: Canvas = set()
    builder = _BUILDERS.get(ch.upper(), _BUILDERS['?'])
    width = int(builder(canvas))
    return canvas, width


def double_vertical_legs(points: set[tuple[int, int]], width: int, *, min_run: int = 5) -> tuple[set[tuple[int, int]], int]:
    """Compatibility hook; stems are now drawn thick in the glyph primitives."""
    return set(points), int(width)


def glyph_to_columns(ch: str, *, vertical_flip: bool = True, column_scale: int = 1, double_legs: bool = True) -> list[int]:
    points, width = glyph_points(ch)
    if double_legs:
        points, width = double_vertical_legs(points, width)
    columns: list[int] = []
    scale = max(1, int(column_scale))
    for x in range(width):
        mask = 0
        for px, visual_row in points:
            if px != x:
                continue
            bit = HEIGHT - 1 - visual_row if vertical_flip else visual_row
            mask |= 1 << bit
        columns.extend([mask] * scale)
    return columns


def message_to_masks(message: str, *, blank_columns: int = 1, vertical_flip: bool = True, column_scale: int = 1, double_legs: bool = True) -> list[int]:
    masks: list[int] = []
    for ch in message:
        masks.extend(glyph_to_columns(ch, vertical_flip=vertical_flip, column_scale=column_scale, double_legs=double_legs))
        masks.extend([0] * max(0, int(blank_columns)))
    while masks and masks[-1] == 0:
        masks.pop()
    return masks

# ---------------------------------------------------------------------------
# Controller-style 16-row bitmap override.
# The segment builder above is kept for history, but rendering uses these
# explicit glyphs because the Videojet preview is a dot-matrix bitmap font.
FONT_VERSION = "videojet-like-16-high-bitmap-v13"

_BITMAPS: dict[str, list[str]] = {
    'A': [
        '.#####.', '.#####.', '##...##', '##...##', '##...##', '##...##',
        '#######', '#######', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '##...##', '##...##'],
    'B': [
        '######.', '######.', '##...##', '##...##', '##...##', '##...##',
        '######.', '######.', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '######.', '######.'],
    'C': [
        '.######', '.######', '##.....', '##.....', '##.....', '##.....',
        '##.....', '##.....', '##.....', '##.....', '##.....', '##.....',
        '##.....', '##.....', '.######', '.######'],
    'D': [
        '######.', '######.', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '######.', '######.'],
    'E': [
        '#######', '#######', '##.....', '##.....', '##.....', '##.....',
        '######.', '######.', '##.....', '##.....', '##.....', '##.....',
        '##.....', '##.....', '#######', '#######'],
    'F': [
        '#######', '#######', '##.....', '##.....', '##.....', '##.....',
        '######.', '######.', '##.....', '##.....', '##.....', '##.....',
        '##.....', '##.....', '##.....', '##.....'],
    'G': [
        '.######', '.######', '##.....', '##.....', '##.....', '##.....',
        '##.....', '##.####', '##.####', '##...##', '##...##', '##...##',
        '##...##', '##...##', '.#####.', '.#####.'],
    'H': [
        '##...##', '##...##', '##...##', '##...##', '##...##', '##...##',
        '#######', '#######', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '##...##', '##...##'],
    'I': [
        '######.', '######.', '..##...', '..##...', '..##...', '..##...',
        '..##...', '..##...', '..##...', '..##...', '..##...', '..##...',
        '..##...', '..##...', '######.', '######.'],
    'J': [
        '..#####', '..#####', '....##.', '....##.', '....##.', '....##.',
        '....##.', '....##.', '....##.', '....##.', '....##.', '##..##.',
        '##..##.', '##..##.', '.####..', '.####..'],
    'K': [
        '##....#', '##...##', '##..##.', '##.##..', '####...', '###....',
        '###....', '###....', '###....', '###....', '###....', '####...',
        '##.##..', '##..##.', '##...##', '##....#'],
    'L': [
        '##.....', '##.....', '##.....', '##.....', '##.....', '##.....',
        '##.....', '##.....', '##.....', '##.....', '##.....', '##.....',
        '##.....', '##.....', '#######', '#######'],
    'M': [
        '##...##', '###.###', '#######', '#######', '##.#.##', '##.#.##',
        '##.#.##', '##...##', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '##...##', '##...##'],
    'N': [
        '##...##', '###..##', '###..##', '####.##', '####.##', '##.####',
        '##.####', '##..###', '##..###', '##...##', '##...##', '##...##',
        '##...##', '##...##', '##...##', '##...##'],
    'O': [
        '.#####.', '.#####.', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '.#####.', '.#####.'],
    'P': [
        '######.', '######.', '##...##', '##...##', '##...##', '##...##',
        '######.', '######.', '##.....', '##.....', '##.....', '##.....',
        '##.....', '##.....', '##.....', '##.....'],
    'Q': [
        '.#####.', '.#####.', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '##...##', '##...##', '##.####', '##..##.',
        '##..###', '##...##', '.######', '.######'],
    'R': [
        '######.', '######.', '##...##', '##...##', '##...##', '##...##',
        '######.', '######.', '##.##..', '##..##.', '##...##', '##...##',
        '##...##', '##...##', '##...##', '##...##'],
    'S': [
        '.######', '.######', '##.....', '##.....', '##.....', '##.....',
        '.#####.', '.#####.', '.....##', '.....##', '.....##', '.....##',
        '.....##', '.....##', '######.', '######.'],
    'T': [
        '########', '########', '...##...', '...##...', '...##...', '...##...',
        '...##...', '...##...', '...##...', '...##...', '...##...', '...##...',
        '...##...', '...##...', '...##...', '...##...'],
    'U': [
        '##...##', '##...##', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '.#####.', '.#####.'],
    'V': [
        '##...##', '##...##', '##...##', '##...##', '##...##', '.##.##.',
        '.##.##.', '.##.##.', '.##.##.', '..###..', '..###..', '..###..',
        '..###..', '...#...', '...#...', '...#...'],
    'W': [
        '##...##', '##...##', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##.#.##', '##.#.##', '##.#.##', '##.#.##', '#######',
        '#######', '###.###', '##...##', '##...##'],
    'X': [
        '##...##', '##...##', '.##.##.', '.##.##.', '..###..', '..###..',
        '...#...', '..###..', '..###..', '.##.##.', '.##.##.', '##...##',
        '##...##', '##...##', '##...##', '##...##'],
    'Y': [
        '##...##', '##...##', '.##.##.', '.##.##.', '..###..', '..###..',
        '..###..', '..###..', '..###..', '..###..', '..###..', '..###..',
        '..###..', '..###..', '..###..', '..###..'],
    'Z': [
        '#######', '#######', '.....##', '....##.', '....##.', '...##..',
        '...##..', '..##...', '..##...', '.##....', '.##....', '##.....',
        '##.....', '##.....', '#######', '#######'],
    '0': [
        '.#####.', '.#####.', '##...##', '##..###', '##.####', '####.##',
        '###..##', '##...##', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '.#####.', '.#####.'],
    '1': [
        '..##...', '.###...', '####...', '..##...', '..##...', '..##...',
        '..##...', '..##...', '..##...', '..##...', '..##...', '..##...',
        '..##...', '..##...', '######.', '######.'],
    '2': [
        '.#####.', '.#####.', '##...##', '.....##', '.....##', '....##.',
        '....##.', '...##..', '..##...', '.##....', '##.....', '##.....',
        '##.....', '##.....', '#######', '#######'],
    '3': [
        '######.', '######.', '.....##', '.....##', '.....##', '....##.',
        '..####.', '..####.', '....##.', '.....##', '.....##', '.....##',
        '.....##', '.....##', '######.', '######.'],
    '4': [
        '....##.', '...###.', '..####.', '.##.##.', '##..##.', '##..##.',
        '#######', '#######', '....##.', '....##.', '....##.', '....##.',
        '....##.', '....##.', '....##.', '....##.'],
    '5': [
        '#######', '#######', '##.....', '##.....', '##.....', '##.....',
        '######.', '######.', '.....##', '.....##', '.....##', '.....##',
        '.....##', '.....##', '######.', '######.'],
    '6': [
        '.#####.', '.#####.', '##.....', '##.....', '##.....', '##.....',
        '######.', '######.', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '.#####.', '.#####.'],
    '7': [
        '#######', '#######', '.....##', '....##.', '....##.', '...##..',
        '...##..', '..##...', '..##...', '.##....', '.##....', '.##....',
        '.##....', '.##....', '.##....', '.##....'],
    '8': [
        '.#####.', '.#####.', '##...##', '##...##', '##...##', '##...##',
        '.#####.', '.#####.', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '.#####.', '.#####.'],
    '9': [
        '.#####.', '.#####.', '##...##', '##...##', '##...##', '##...##',
        '##...##', '##...##', '.######', '.######', '.....##', '.....##',
        '.....##', '.....##', '.#####.', '.#####.'],
    ' ': ['.......'] * 16,
    '?': [
        '.#####.', '.#####.', '##...##', '.....##', '.....##', '....##.',
        '...##..', '..##...', '..##...', '..##...', '.......', '.......',
        '..##...', '..##...', '..##...', '.......'],
}


def glyph_points(ch: str) -> tuple[set[tuple[int, int]], int]:  # type: ignore[override]
    rows = _BITMAPS.get(ch.upper(), _BITMAPS['?'])
    points: set[tuple[int, int]] = set()
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            if value not in {'.', '0', ' '}:
                points.add((x, y))
    width = len(rows[0]) if rows else 0
    return points, width


def glyph_to_columns(ch: str, *, vertical_flip: bool = True, column_scale: int = 1, double_legs: bool = True) -> list[int]:  # type: ignore[override]
    points, width = glyph_points(ch)
    columns: list[int] = []
    scale = max(1, int(column_scale))
    for x in range(width):
        mask = 0
        for px, visual_row in points:
            if px != x:
                continue
            bit = HEIGHT - 1 - visual_row if vertical_flip else visual_row
            mask |= 1 << bit
        columns.extend([mask] * scale)
    return columns


def message_to_masks(message: str, *, blank_columns: int = 1, vertical_flip: bool = True, column_scale: int = 1, double_legs: bool = True) -> list[int]:  # type: ignore[override]
    masks: list[int] = []
    for ch in message:
        masks.extend(glyph_to_columns(ch, vertical_flip=vertical_flip, column_scale=column_scale, double_legs=double_legs))
        masks.extend([0] * max(0, int(blank_columns)))
    while masks and masks[-1] == 0:
        masks.pop()
    return masks


# Real Videojet BDF font support.  These definitions intentionally override
# the handmade fallback functions above when the converted ABF font is present.
from pathlib import Path as _Path
import os as _os

_FONT_DIR = _Path(__file__).resolve().parents[2] / "inputs" / "fonts"
_BDF_FONT_VARIANTS = {
    "original": _FONT_DIR / "videojet_16_high_international.bdf",
}
_BDF_FONT_VARIANT = _os.environ.get("VIDEOJET_FONT_VARIANT", "original")
if _BDF_FONT_VARIANT not in _BDF_FONT_VARIANTS:
    _BDF_FONT_VARIANT = "original"
_BDF_FONT_PATH = _BDF_FONT_VARIANTS[_BDF_FONT_VARIANT]


def _parse_bdf_font(path: _Path):
    glyphs: dict[int, dict] = {}
    ascent = HEIGHT
    descent = 0
    current: dict | None = None
    in_bitmap = False
    bitmap_rows: list[str] = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("FONT_ASCENT"):
            ascent = int(line.split()[1])
            continue
        if line.startswith("FONT_DESCENT"):
            descent = int(line.split()[1])
            continue
        if line.startswith("STARTCHAR"):
            current = {"name": line.split(maxsplit=1)[1], "encoding": None, "dwidth": 0, "bbx": (0, 0, 0, 0)}
            bitmap_rows = []
            in_bitmap = False
            continue
        if current is None:
            continue
        if line.startswith("ENCODING"):
            current["encoding"] = int(line.split()[1])
        elif line.startswith("DWIDTH"):
            current["dwidth"] = int(line.split()[1])
        elif line.startswith("BBX"):
            _tag, w, h, xoff, yoff = line.split()
            current["bbx"] = (int(w), int(h), int(xoff), int(yoff))
        elif line == "BITMAP":
            in_bitmap = True
            bitmap_rows = []
        elif line == "ENDCHAR":
            encoding = current.get("encoding")
            if encoding is not None and int(encoding) >= 0:
                bbx_w, bbx_h, xoff, yoff = current["bbx"]
                points: set[tuple[int, int]] = set()
                row_hex_width = max(1, (bbx_w + 7) // 8 * 2)
                for row_index, hex_row in enumerate(bitmap_rows[:bbx_h]):
                    bits = bin(int(hex_row or "0", 16))[2:].zfill(row_hex_width * 4)
                    for bit_index, bit in enumerate(bits[:bbx_w]):
                        if bit != "1":
                            continue
                        glyph_x = xoff + bit_index
                        glyph_y = yoff + (bbx_h - 1 - row_index)
                        visual_row = ascent - 1 - glyph_y
                        if 0 <= visual_row < HEIGHT:
                            points.add((glyph_x, visual_row))
                if points:
                    min_x = min(x for x, _y in points)
                    if min_x < 0:
                        points = {(x - min_x, y) for x, y in points}
                width = max(int(current.get("dwidth", 0) or 0), xoff + bbx_w, 1)
                if points:
                    width = max(width, max(x for x, _y in points) + 1)
                glyphs[int(encoding)] = {"points": points, "width": int(width)}
            current = None
            in_bitmap = False
            bitmap_rows = []
        elif in_bitmap:
            bitmap_rows.append(line)
    return {"glyphs": glyphs, "ascent": ascent, "descent": descent}


def available_font_variants() -> dict[str, str]:
    return {name: str(path) for name, path in _BDF_FONT_VARIANTS.items() if path.exists()}


def _load_bdf_variant(variant: str):
    variant = str(variant)
    if variant not in _BDF_FONT_VARIANTS:
        raise ValueError(f"unknown font variant {variant!r}; choose {sorted(_BDF_FONT_VARIANTS)}")
    path = _BDF_FONT_VARIANTS[variant]
    return (_parse_bdf_font(path) if path.exists() else None), path


def set_font_variant(variant: str) -> str:
    global _BDF_FONT, _BDF_FONT_PATH, _BDF_FONT_VARIANT, FONT_VERSION, FONT_SOURCE, FONT_SOURCE_PATH
    font, path = _load_bdf_variant(variant)
    _BDF_FONT = font
    _BDF_FONT_PATH = path
    _BDF_FONT_VARIANT = str(variant)
    if _BDF_FONT is not None:
        FONT_VERSION = "videojet-bdf-16-high-international-original-v1"
        FONT_SOURCE = f"bdf:{_BDF_FONT_VARIANT}"
        FONT_SOURCE_PATH = str(_BDF_FONT_PATH)
    else:
        FONT_VERSION = "videojet-like-16-high-bitmap-v13"
        FONT_SOURCE = "handmade-fallback"
        FONT_SOURCE_PATH = ""
    return FONT_VERSION


try:
    _BDF_FONT, _BDF_FONT_PATH = _load_bdf_variant(_BDF_FONT_VARIANT)
except Exception:
    _BDF_FONT = None

if _BDF_FONT is not None:
    FONT_VERSION = "videojet-bdf-16-high-international-original-v1"
    FONT_SOURCE = f"bdf:{_BDF_FONT_VARIANT}"
    FONT_SOURCE_PATH = str(_BDF_FONT_PATH)

    def glyph_points(ch: str) -> tuple[set[tuple[int, int]], int]:  # type: ignore[override]
        codepoint = ord(ch[0]) if ch else 32
        glyph = _BDF_FONT["glyphs"].get(codepoint) or _BDF_FONT["glyphs"].get(ord(ch.upper()[0])) if ch else None
        if glyph is None:
            glyph = _BDF_FONT["glyphs"].get(ord("?")) or _BDF_FONT["glyphs"].get(32)
        if glyph is None:
            return set(), 1
        return set(glyph["points"]), int(glyph["width"])

    def glyph_to_columns(ch: str, *, vertical_flip: bool = True, column_scale: int = 1, double_legs: bool = True) -> list[int]:  # type: ignore[override]
        points, width = glyph_points(ch)
        columns: list[int] = []
        scale = max(1, int(column_scale))
        for x in range(width):
            mask = 0
            for px, visual_row in points:
                if px != x:
                    continue
                bit = HEIGHT - 1 - visual_row if vertical_flip else visual_row
                mask |= 1 << bit
            columns.extend([mask] * scale)
        return columns

    def message_to_masks(message: str, *, blank_columns: int = 1, vertical_flip: bool = True, column_scale: int = 1, double_legs: bool = True) -> list[int]:  # type: ignore[override]
        masks: list[int] = []
        for ch in message:
            masks.extend(glyph_to_columns(ch, vertical_flip=vertical_flip, column_scale=column_scale, double_legs=double_legs))
            masks.extend([0] * max(0, int(blank_columns)))
        while masks and masks[-1] == 0:
            masks.pop()
        return masks
