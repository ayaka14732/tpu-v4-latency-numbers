"""顺序测量 16 个 BC 与四颗芯片 TC0 的 LCC/GTC 计数比例，核对完整性并生成 results/15_bc_clock.md。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

TC_SLOPE = 32 / 3
GTC_TICKS_PER_NS = 11.2
WORKLOADS = ('spin', 'dma', 'host_wait')

def run(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    args.archive.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(args.libtpu_root)
    for device in range(16):
        for outer in ('gtc', 'lcc'):
            name = f'bc{device:02d}_{outer}_outer'
            output = args.output / name
            if not (output / 'summary.json').exists():
                assert not args.analyze_only, f'缺少已完成记录：{output}'
                output.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(here / '15_bc_clock.py'), '--device', str(device), '--outer', outer, '--seed', str(1500 + device), '--output', str(output)]
                print('starting', name, flush=True)
                with (output / 'run.log').open('w') as stream:
                    subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            shutil.copyfile(output / 'summary.json', args.archive / f'{name}.json')
    shutil.copyfile(args.output / 'bc00_gtc_outer' / 'program.bca', args.archive / 'gtc_outer.bca')
    shutil.copyfile(args.output / 'bc00_lcc_outer' / 'program.bca', args.archive / 'lcc_outer.bca')
    # TC 对照使用共享环境的 0.0.49 与相邻 tpuasm；两种 runtime 的 TPU 进程不并行。
    output = args.output / 'tc'
    if not (output / 'summary.json').exists():
        assert not args.analyze_only, f'缺少已完成记录：{output}'
        output.mkdir(parents=True, exist_ok=True)
        environment['PYTHONPATH'] = str(here.parents[1] / 'tpuasm' / 'src')
        print('starting tc', flush=True)
        with (output / 'run.log').open('w') as stream:
            subprocess.run([sys.executable, str(here / '15_tc_clock.py'), '--output', str(output)], env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    shutil.copyfile(output / 'summary.json', args.archive / 'tc.json')
    for outer in ('gtc', 'lcc'):
        shutil.copyfile(output / f'{outer}_outer.full.tpuasm', args.archive / f'tc_{outer}_outer.full.tpuasm')

def intervals(document: dict) -> tuple[dict[str, list[tuple[int, int, int]]], list[int]]:
    """返回每类负载的 (ΔLCC, ΔGTC, host ns) 与连续 GTC 读取的逐周期增量；并核对 dense LCC 与 spin 的精确周期。"""
    outer = document['outer']
    result = {name: [] for name in WORKLOADS}
    steps = []
    for row in document['records']:
        words = row['words']
        if row['mode'] == 'dense_lcc':
            assert [b - a for a, b in zip(words, words[1:])] == [1] * 5
            continue
        if row['mode'] == 'dense_gtc':
            steps.append([b - a for a, b in zip(words, words[1:])])
            continue
        outer_delta, inner_delta = words[1] - words[0], words[3] - words[2]
        lcc, gtc = (inner_delta, outer_delta) if outer == 'gtc' else (outer_delta, inner_delta)
        # 内外层读数各相隔一个 bundle：GTC 在外时 GTC 区间比 LCC 区间多 2 cycles，LCC 在外时少 2 cycles。
        lcc_between_gtc = lcc + 2 if outer == 'gtc' else lcc - 2
        if row['mode'] == 'spin':
            assert lcc == 4 * row['count'] + (1 if outer == 'gtc' else 3)
        result[row['mode']].append((lcc_between_gtc, gtc, row['host_ns']))
    return result, steps

def ppm(points: list[tuple[int, int, int]]) -> float:
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    slope = np.polyfit(x, y, 1)[0]
    return (slope / TC_SLOPE - 1) * 1e6

def tc_rows(args: argparse.Namespace) -> tuple[list[str], dict[str, list[tuple[int, int, int]]]]:
    """同芯片 TC0 对照：端点依次为外层 BEGIN、内层 BEGIN、内层 END、外层 END。"""
    document = json.loads((args.archive / 'tc.json').read_text())
    assert document['protocol'] == 'tc-lcc-gtc-vdelay-v1' and document['libtpu'] == '0.0.49'
    lines, chips = [], {}
    for device in sorted({row['device'] for row in document['records']}):
        rows = [row for row in document['records'] if row['device'] == device]
        points, offsets = [], set()
        for row in rows:
            w = row['words']
            outer_delta, inner_delta = w[3] - w[0], w[2] - w[1]
            lcc, gtc = (inner_delta, outer_delta) if row['outer'] == 'gtc' else (outer_delta, inner_delta)
            offsets.add(lcc - row['delay'] - (0 if row['outer'] == 'gtc' else 2))
            points.append((lcc + 2 if row['outer'] == 'gtc' else lcc - 2, gtc, 0))
        residual = [p[1] - TC_SLOPE * p[0] for p in points]
        chip = ','.join(str(c) for c in rows[0]['coords'][:2])
        chips[chip] = points
        cells = [str(device), f'({chip})', str(len(points)), f'{min(offsets)}–{max(offsets)}', f'{ppm(points):+.2f}', f'[{min(residual):.1f}, {max(residual):.1f}]', f'{max(p[0] for p in points):,}']
        lines.append('| ' + ' | '.join(cells) + ' |')
    return lines, chips

def spread(points: list[tuple[int, int, int]], detrend: bool) -> str:
    """残差范围；detrend 时先扣除该组自身的拟合直线，只留相位抖动。"""
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    residual = y - np.polyval(np.polyfit(x, y, 1), x) if detrend else y - TC_SLOPE * x
    return f'[{residual.min():.0f}, {residual.max():.0f}]'

def analyze(args: argparse.Namespace) -> None:
    lines = []
    chips: dict[str, list[float]] = {}
    chip_points: dict[str, list[tuple[int, int, int]]] = {}
    totals = dict(records=0, intervals=0)
    overall, windows, increments = [], [], []
    for device in range(16):
        for outer in ('gtc', 'lcc'):
            document = json.loads((args.archive / f'bc{device:02d}_{outer}_outer.json').read_text())
            assert document['protocol'] == 'bc-lcc-gtc-nested-v1' and document['libtpu'] == '0.0.46' and document['cleanup_clean']
            data, steps = intervals(document)
            windows += [sum(row[i:i + 3]) for row in steps for i in range(len(row) - 2)]
            increments += [step for row in steps for step in row]
            identity = document['device']
            points = [p for name in WORKLOADS for p in data[name]]
            residual = [p[1] - TC_SLOPE * p[0] for p in points]
            host = np.polyfit([p[1] / GTC_TICKS_PER_NS for p in data['host_wait']], [p[2] for p in data['host_wait']], 1)[0]
            longest = max(p[0] for p in points)
            overall.append(ppm(points))
            chips.setdefault(','.join(str(c) for c in identity['chip_coordinates'][:2]), []).append(ppm(points))
            chip_points.setdefault(','.join(str(c) for c in identity['chip_coordinates'][:2]), []).extend(points)
            totals['records'] += len(document['records'])
            totals['intervals'] += len(points)
            chip = ','.join(str(c) for c in identity['chip_coordinates'][:2])
            cells = [str(identity['wrapper_id']), f'({chip})', str(identity['core_id']), outer.upper(), str(len(points))]
            cells += [f'{ppm(data[name]):+.2f}' for name in WORKLOADS]
            cells += [f'{ppm(points):+.2f}', f'[{min(residual):.1f}, {max(residual):.1f}]', f'{longest:,}', f'{host:.6f}']
            lines.append('| ' + ' | '.join(cells) + ' |')
    note = (
        'BC 本地 LCC 与 GTC 的计数比例。每个 BC 运行两种嵌套顺序（GTC 在外／LCC 在外），区间内依次为空循环 spin、N 次 HBM→BC 私有 BMEM DMA 加 done/fence，以及 `swait.gt` 等待 host 写 sflag 的 host_wait。'
        'ppm 列为 ΔGTC 对 ΔLCC 的拟合斜率相对 TC 的 32/3 的偏差；ΔLCC 已按内外层相差一个 bundle 的端点换算到两个 GTC 采样点之间。残差为每个样本的 ΔGTC−(32/3)ΔLCC，单位为 raw GTC tick。'
        'host 列为 host 完成时间对 ΔGTC/11.2 的拟合斜率，只作 GTC 时基的粗核对。[方法与结论](../README.md#12-bc-本地周期与-tc-周期)。'
    )
    header = '| BC wrapper | 芯片 | BC core | 外层 | 区间数 | spin ppm | DMA ppm | host_wait ppm | 全部 ppm | 残差 ticks | 最长 ΔLCC | host ns／(ΔGTC/11.2) |'
    rule = '| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |'
    summary = f'合计 {totals["records"]} 次正式请求：{totals["intervals"]} 个嵌套区间、{totals["records"] - totals["intervals"]} 组逐 bundle 连续读取。32 组拟合的全部 ppm 落在 [{min(overall):+.2f}, {max(overall):+.2f}]。'
    outside = sorted(step for step in increments if step not in (1, 15, 16))
    summary += f'连续 GTC 读取共 {len(increments)} 个相邻增量，{len(increments) - len(outside)} 个属于 1、15、16，其余为 {outside}；{len(windows)} 个重叠三周期窗口中 {windows.count(32)} 个合计 32 ticks。'
    tc_lines, tc_chips = tc_rows(args)
    tc_note = '同芯片对照：每颗芯片的 TC0 在 tpuasm 插入的机器程序中嵌套读取 LCC 与 GTC，中间以 `vdelay` 加 `sfence` 拉长区间，两种嵌套顺序各测；ΔLCC−H 列为扣除 vdelay 长度 H 后的固定周期（LCC 在外时另扣两端各一个 bundle）。'
    tc_header = '| JAX device | 芯片 | 区间数 | ΔLCC−H | 全部 ppm | 残差 ticks | 最长 ΔLCC |'
    tc_rule = '| ---: | --- | ---: | --- | ---: | --- | ---: |'
    pairs = '；'.join(f'({chip}) BC {min(chips[chip]):+.2f}…{max(chips[chip]):+.2f}，TC0 {ppm(tc_chips[chip]):+.2f}' for chip in sorted(tc_chips))
    summary += f'按芯片比较 ppm：{pairs}。'
    chip_note = '按芯片合并：BC 列合并该芯片 4 个 BC 的两种嵌套顺序。短区间残差只取 ΔLCC<10⁴ 的样本，按 32/3 计算；去趋势残差先扣除各组自身的拟合直线，只剩 GTC 相对 LCC 的相位抖动。单位均为 raw GTC tick。'
    chip_header = '| 芯片 | BC ppm | TC0 ppm | BC−TC0 ppm | BC 短区间残差 | BC 去趋势残差 | TC0 去趋势残差 |'
    chip_rule = '| --- | ---: | ---: | ---: | --- | --- | --- |'
    chip_lines = []
    for chip in sorted(tc_chips):
        bc, tc = chip_points[chip], tc_chips[chip]
        short = [p for p in bc if p[0] < 10**4]
        cells = [f'({chip})', f'{ppm(bc):+.3f}', f'{ppm(tc):+.3f}', f'{ppm(bc) - ppm(tc):+.3f}', spread(short, False), spread(bc, True), spread(tc, True)]
        chip_lines.append('| ' + ' | '.join(cells) + ' |')
    text = note + '\n\n' + header + '\n' + rule + '\n' + '\n'.join(lines) + '\n\n' + tc_note + '\n\n' + tc_header + '\n' + tc_rule + '\n' + '\n'.join(tc_lines) + '\n\n' + chip_note + '\n\n' + chip_header + '\n' + chip_rule + '\n' + '\n'.join(chip_lines) + '\n\n' + summary + '\n'
    path = Path(__file__).resolve().parents[1] / 'results' / '15_bc_clock.md'
    path.write_text(text)
    print(summary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/15_bc_clock_suite'))
    parser.add_argument('--archive', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/15_bc_clock'))
    parser.add_argument('--libtpu-root', type=Path, default=Path('/tmp/libtpu-0.0.46'))
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    if not args.analyze_only:
        run(args)
    analyze(args)
