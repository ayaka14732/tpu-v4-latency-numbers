"""生成一张可单独传播的英文周期公式总结图；经验公式直接取自 results/ 中的冻结分表。"""

from __future__ import annotations

import argparse
from html import escape
import math
from pathlib import Path
import re
import subprocess

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

RESULTS = Path(__file__).resolve().parents[1] / 'results'
WIDTH = 1200
LEFT = 48
FORMULA_X = 380
CHART = (620, 1152)
DECADES = (1, 6)
ROW = 34
GROUP = 40
FONT_URL = 'https://mirrors.ctan.org/fonts/lm-math/opentype/latinmodern-math.otf'
FONT_PATH = Path('/tmp/tpu_latency_numbers/fonts/latinmodern-math.otf')
SPACING = {'bin': 4 / 18, 'rel': 5 / 18}
RELATION = {True: '=', False: '≈'}
CLASS = {True: 'exact', False: 'approx'}
CLASSES = {'+': 'bin', '=': 'rel', '≥': 'rel', '≈': 'rel'}

def center(name: str, *keys: str) -> tuple[float, float]:
    """读取分表中首列依次等于 keys 的一行，返回 `a + bK` 的系数。"""
    for line in (RESULTS / f'{name}.md').read_text().splitlines():
        cells = [cell.strip().strip('`') for cell in line.strip('|').split('|')]
        if tuple(cells[:len(keys)]) == keys:
            match = re.search(r'([\d.]+) \+ ([\d.]+)(?: \* \(S / 1024\)|K)', line)
            return float(match[1]), float(match[2])
    raise KeyError((name, keys))

def bc_to_tc() -> float:
    """第 5、7、9 节在芯片 (0,0) 的 BC0（wrapper 0）上计时；按同芯片 BC 与 TC0 的 ΔGTC/ΔLCC 之比换算为 TC 周期。"""
    rows = [[cell.strip() for cell in line.strip('|').split('|')] for line in (RESULTS / '15_bc_clock.md').read_text().splitlines() if line.startswith('| ')]
    bc = [float(row[8]) for row in rows if len(row) == 12 and row[0] == '0' and row[1] == '(0,0)']
    (tc,) = [float(row[4]) for row in rows if len(row) == 7 and row[1] == '(0,0)']
    assert len(bc) == 2
    return (1 + sum(bc) / len(bc) * 1e-6) / (1 + tc * 1e-6)

Row = tuple[str, float, float, bool, str, str]

def exact(label: str, a: int, b: float, formula: str) -> Row:
    """返回 (路径, 4 KiB 周期, 1 MiB 周期, 是否为逐周期成立的整数公式, 公式, 公式条件)。"""
    return label, a + 4 * b, a + 1024 * b, True, formula, ''

def measured(label: str, coefficients: tuple[float, float]) -> Row:
    """先冻结、再经独立尺寸验证的中心公式。"""
    a, b = coefficients
    return label, a + 4 * b, a + 1024 * b, False, f'{a:,.0f} + {b:.2f}K' if b < 10 else f'{a:,.0f} + {b:.1f}K', ''

def groups() -> list[tuple[str, list[Row]]]:
    return [
        ('Into TC VREG', [
            exact('TC VMEM → TC VREG', 13, 1 / 4, '13 + K/4'),
            # 第 3.4 节 D=27 深流水：C = 2N + 66 仅对 N ≥ 27 成立；单个向量无法流水，为 67 cycles。
            ('CMEM → TC VREG', 67, 66 + 1024 / 2, True, '66 + K/2', 'K ≥ 108'),
        ]),
        ('On-chip DMA', [
            exact('CMEM → TC VMEM', 311, 1 / 2, '311 + K/2'),
            exact('TC VMEM → CMEM', 309, 1, '309 + K'),
            measured('CMEM → HBM', center('02_local', 'cmem_hbm', '1 / 0', '1')),
            measured('TC VMEM → HBM', center('02_local', 'vmem_hbm', '1 / 0', '1')),
            measured('HBM → CMEM', center('02_local', 'hbm_cmem', '1 / 0', '1')),
            measured('HBM → TC VMEM', center('02_local', 'hbm_vmem', '1 / 0', '1')),
            measured('TC0 VMEM → TC1 VMEM', center('04_remote', 'same_chip_0_w1', '1', '0')),
        ]),
        ('Cross-chip ICI', [
            measured('TC VMEM → TC VMEM, 1 hop', center('04_remote', 'c0_c1_w1', '1', '0')),
            measured('TC VMEM → TC VMEM, 2 hops', center('04_remote', 'c0_c3_w1', '1', '0')),
            measured('CMEM → CMEM, 1 hop', center('04_cmem', 'c0_c1_w1', '1', '0')),
        ]),
        ('BarnaCore', [
            measured('HBM → BC BMEM', tuple(value * bc_to_tc() for value in center('05_bc', '1'))),
            measured('TC VMEM → BC BMEM', tuple(value * bc_to_tc() for value in center('07_bc_pull', 'vmem', '1'))),
            measured('CMEM → BC BMEM', tuple(value * bc_to_tc() for value in center('07_bc_pull', 'cmem', '1'))),
        ]),
        ('Host', [
            measured('HBM → Host, TC-initiated', center('06_host', '1')),
            measured('HBM → Host, Host-initiated', tuple(value * bc_to_tc() for value in center('09_magic_host', '1'))),
        ]),
    ]

class MathText:
    """用 Latin Modern Math 的字形轮廓排版 TeX 子集：数字与标点直立，字母为数学斜体，`^{...}` 为上标，`\\text{...}` 为直立正文；二元运算符与关系符两侧按 TeX 留 4mu／5mu。"""

    def __init__(self) -> None:
        if not FONT_PATH.exists():
            FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['curl', '-sSLf', '-o', FONT_PATH, FONT_URL], check=True)
        self.font = TTFont(FONT_PATH)
        self.cmap = self.font.getBestCmap()
        self.paths: dict[str, str] = {}
        constants = self.font['MATH'].table.MathConstants
        self.script = constants.ScriptPercentScaleDown / 100
        self.shift = constants.SuperscriptShiftUp.Value / 1000

    def glyph(self, code: int) -> tuple[str, float]:
        name = self.cmap[code]
        if name not in self.paths:
            pen = SVGPathPen(self.font.getGlyphSet())
            self.font.getGlyphSet()[name].draw(pen)
            self.paths[name] = pen.getCommands()
        return name, self.font['hmtx'][name][0] / 1000

    def layout(self, tex: str) -> tuple[list[tuple[str, float, float, float]], float]:
        """返回以 em 为单位的 (字形, x, 基线上移, 缩放) 和总宽度。"""
        placed, x, gap, previous, index = [], 0.0, 0.0, None, 0
        while index < len(tex):
            if tex.startswith(('^{', '\\text{'), index):
                x, gap = x + gap, 0.0
                close = tex.index('}', index)
                superscript = tex[index] == '^'
                for char in tex[tex.index('{', index) + 1:close]:
                    name, advance = self.glyph(ord(char))
                    size = self.script if superscript else 1
                    placed.append((name, x, self.shift if superscript else 0, size))
                    x += advance * size
                index, previous = close + 1, 'ord'
                continue
            char, index = tex[index], index + 1
            if char == ' ':
                continue
            kind = CLASSES.get(char, 'ord')
            x += SPACING[kind] if kind != 'ord' and previous is not None else gap
            code = ord(char)
            if char.isascii() and char.isalpha():
                code = 0x210E if char == 'h' else (0x1D434 + code - ord('A') if char.isupper() else 0x1D44E + code - ord('a'))
            name, advance = self.glyph(code)
            placed.append((name, x, 0, 1))
            x, gap, previous = x + advance, SPACING.get(kind, 0.0), kind
        return placed, x

    def place(self, x: float, y: float, tex: str, size: float, cls: str, anchor: str = 'start') -> tuple[str, float]:
        """把 tex 放在基线 y 上，返回 SVG 片段和以像素计的宽度。"""
        placed, width = self.layout(tex)
        x -= {'start': 0, 'middle': width / 2, 'end': width}[anchor] * size
        uses = [f'<use href="#lm-{name}" transform="translate({x + dx * size:.2f} {y - dy * size:.2f}) scale({size * k / 1000:.5f} {-size * k / 1000:.5f})"/>' for name, dx, dy, k in placed if self.paths[name]]
        return f'<g class="{cls}">{"".join(uses)}</g>', width * size

    def defs(self) -> str:
        return '<defs>' + ''.join(f'<path id="lm-{name}" d="{d}"/>' for name, d in self.paths.items() if d) + '</defs>'

def scale(cycles: float) -> float:
    low, high = DECADES
    return CHART[0] + (math.log10(cycles) - low) / (high - low) * (CHART[1] - CHART[0])

def text(x: float, y: float, content: str, cls: str, anchor: str = 'start') -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(content)}</text>'

def render() -> str:
    math_text = MathText()
    body = [
        text(LEFT, 72, 'Latency Numbers Every TPU Programmer Should Know', 'title'),
    ]
    for power in range(DECADES[0], DECADES[1] + 1):
        body.append(math_text.place(scale(10 ** power), 122, f'10^{{{power}}}', 15, 'tick', 'middle')[0])
    y = 130
    grid_top = y
    for title, rows in groups():
        y += GROUP
        body.append(text(LEFT, y - 12, title, 'group'))
        body.append(f'<line x1="{LEFT}" x2="{CHART[1]}" y1="{y - 4:.1f}" y2="{y - 4:.1f}" class="rule"/>')
        for label, small, large, is_exact, formula, condition in rows:
            middle = y + ROW / 2
            x0, x1 = scale(small), scale(large)
            relation_svg, relation_width = math_text.place(FORMULA_X, middle + 6, RELATION[is_exact], 20, CLASS[is_exact])
            formula_x = FORMULA_X + relation_width + 20 * SPACING['rel']
            formula_svg, formula_width = math_text.place(formula_x, middle + 6, formula, 20, 'formula')
            body += [
                text(LEFT, middle + 6, label, 'label'),
                relation_svg,
                formula_svg,
                math_text.place(formula_x + formula_width + 12, middle + 6, condition, 15, 'condition')[0],
                math_text.place(x0 - 9, middle + 4.5, f'{small:,.0f}', 14, 'value', 'end')[0],
                math_text.place(x1 + 9, middle + 4.5, f'{large:,.0f}', 14, 'value')[0],
                f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{middle:.1f}" y2="{middle:.1f}" class="bar"/>',
                f'<circle cx="{x0:.1f}" cy="{middle:.1f}" r="5" class="ring"/>',
                f'<circle cx="{x1:.1f}" cy="{middle:.1f}" r="5.5" class="dot"/>',
            ]
            y += ROW
    grid = [f'<line x1="{scale(10 ** p):.1f}" x2="{scale(10 ** p):.1f}" y1="{grid_top}" y2="{y + 4}" class="grid"/>' for p in range(DECADES[0], DECADES[1] + 1)]
    y += 44
    body.append(math_text.place(LEFT, y, '\\text{TPU v4 cycles }C(K) = a + bK\\text{, where }K\\text{ is KiB per transfer; }1\\text{ cycle} ≈ 0.95\\text{ ns}', 16, 'caption')[0])

    def legend(x: float) -> tuple[list[str], float]:
        """从 x 起排列图例，返回 SVG 片段和末尾 x。"""
        parts = []
        for index, (mark, name) in enumerate(((True, 'exact'), (False, 'fitted'), ('ring', '4 KiB'), ('dot', '1 MiB'))):
            x += 24 if index else 0
            if isinstance(mark, str):
                parts.append(f'<circle cx="{x + 5:.1f}" cy="{y - 5}" r="{5 if mark == "ring" else 5.5}" class="{mark}"/>')
                x += 17
            else:
                symbol, width = math_text.place(x, y, RELATION[mark], 16, CLASS[mark])
                parts.append(symbol)
                x += width + 16 * SPACING['rel']
            label, width = math_text.place(x, y, f'\\text{{{name}}}', 16, 'caption')
            parts.append(label)
            x += width
        return parts, x

    body += legend(CHART[1] - legend(0)[1])[0]
    body.append(text(LEFT, y + 26, 'Conditions and residuals: github.com/ayaka14732/tpu-v4-latency-numbers', 'foot'))
    y += 26
    height = y + 40
    style = '''
    .bg { fill: #fcfcfb; }
    text { font-family: Inter, "Noto Sans SC", "PingFang SC", "Microsoft YaHei", "Droid Sans Fallback", system-ui, sans-serif; fill: #0b0b0b; }
    .title { font-size: 36px; font-weight: 700; letter-spacing: -0.5px; }
    .formula { fill: #0b0b0b; }
    .caption, .tick, .value { fill: #52514e; }
    .condition, .approx { fill: #7a7974; }
    .exact { fill: #13875c; }
    .group { font-size: 14px; font-weight: 700; fill: #52514e; }
    .label { font-size: 16px; font-weight: 600; }
    .foot { font-size: 12px; fill: #7a7974; }
    .rule { stroke: #d9d8d2; stroke-width: 1; }
    .grid { stroke: #ecebe6; stroke-width: 1; }
    .bar { stroke: #2a78d6; stroke-width: 2; stroke-linecap: round; }
    circle.ring { fill: #fcfcfb; stroke: #2a78d6; stroke-width: 2; }
    circle.dot { fill: #2a78d6; stroke: #fcfcfb; stroke-width: 2; }
    @media (prefers-color-scheme: dark) {
        .bg { fill: #1a1a19; }
        text { fill: #ffffff; }
        .formula { fill: #ffffff; }
        .caption, .group, .tick, .value { fill: #c3c2b7; }
        .foot, .condition, .approx { fill: #9a998f; }
        .exact { fill: #199e70; }
        .rule { stroke: #3a3a37; }
        .grid { stroke: #2a2a28; }
        .bar { stroke: #3987e5; }
        circle.ring { fill: #1a1a19; stroke: #3987e5; }
        circle.dot { fill: #3987e5; stroke: #1a1a19; }
    }
    '''
    return '\n'.join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Latency Numbers Every TPU Programmer Should Know</title>',
        '<desc id="desc">Measured TPU v4 cycle formulas a + b·K for each data path, with cycles at 4 KiB and 1 MiB.</desc>',
        f'<style>{style}</style>',
        math_text.defs(),
        f'<rect width="{WIDTH}" height="{height}" class="bg"/>',
        *grid,
        *body,
        '</svg>',
        '',
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RESULTS.parent / 'latency_numbers.svg')
    args = parser.parse_args()
    args.output.write_text(render())
    print(args.output)
