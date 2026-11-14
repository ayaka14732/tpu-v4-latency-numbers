"""只用 CPU 重放 16_bc_tpuasm.py 的归档：冻结发现组的中心公式、计算验证组残差、核对 LCC/GTC 读数并生成 results/ 中的表。

`--freeze` 由 `dma_discovery.json`、`pull_discovery.json`、`host_discovery.json` 求出每组的 `a + bK`（K = S/1024，发现尺寸上 24 次中位数的最小二乘直线），写进 `models.json`；文件已存在时只核对重算的结果与它相同，不改写。不带 `--freeze` 时读取冻结的模型与验证组，生成 `results/05_bc.md`、`results/07_bc_pull.md`、`results/09_host_read.md`，若有 `clock.json` 再生成 `results/15_bc_clock.md`。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
TICKS_PER_CYCLE = 32 / 3
GTC_PER_NS = 11.2
SOURCE_NAMES = {'hbm': 'hbm', 'cmem': 'cmem', 'tc0': 'vmem'}

def load(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / f'{name}.json').read_text())

def groups(document: dict[str, Any], family: str) -> dict[str, list[dict[str, Any]]]:
    """按模型分组：dma 与 host 按 W，pull 按（来源，W）；只收全部样本数据正确的配置。"""
    result: dict[str, list[dict[str, Any]]] = {}
    for config in document['configs']:
        assert len(config['records']) == 24, config['size_bytes']
        assert all(row['full_payload_and_guards_correct'] for row in config['warmup'] + config['records']), (family, config['window'], config['size_bytes'])
        assert config.get('finished', True) and config.get('scratch_intact') in (None, True), (family, config['size_bytes'])
        if family == 'host' and config['size_bytes'] == 0:
            continue
        key = f'{SOURCE_NAMES[config["source"]]}/{config["window"]}' if family == 'pull' else f'{config["mode"]}/{config["window"]}' if family == 'host' else str(config['window'])
        result.setdefault(key, []).append(config)
    return result

def errors(configs: list[dict[str, Any]], model: dict[str, float]) -> dict[str, float]:
    residuals, medians, relative = [], [], []
    for config in configs:
        kib = config['size_bytes'] / 1024
        predicted = model['intercept_cycles'] + model['cycles_per_kib'] * kib
        values = np.array([row['cycles'] for row in config['records']], dtype=float) - predicted
        residuals.extend(values.tolist())
        medians.append(float(np.median(values)))
        relative.append(float(np.median(values)) / predicted)
    return {
        'samples': len(residuals),
        'max_median_error_cycles': float(np.max(np.abs(medians))),
        'max_relative_median_error': float(np.max(np.abs(relative))),
        'min_residual_cycles': float(np.min(residuals)),
        'max_residual_cycles': float(np.max(residuals)),
    }

def fit(configs: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.array([config['size_bytes'] / 1024 for config in configs])
    y = np.array([np.median([row['cycles'] for row in config['records']]) for config in configs])
    slope, intercept = np.polyfit(x, y, 1)
    model: dict[str, Any] = {'intercept_cycles': round(float(intercept), 6), 'cycles_per_kib': round(float(slope), 6), 'discovery_kib': [int(v) for v in x]}
    model['discovery'] = errors(configs, model)
    return model

def freeze(root: Path) -> None:
    models = {family: {key: fit(configs) for key, configs in sorted(groups(load(root, f'{family}_discovery'), family).items())} for family in ('dma', 'pull', 'host')}
    path = root / 'models.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(models)), 'the discovery data changed; frozen models must not be overwritten'
        print('models.json already frozen and reproduced')
    else:
        path.write_text(json.dumps(models, indent=2) + '\n')
        print(f'froze {sum(len(v) for v in models.values())} models')
    gaps = load(root, 'dma_discovery')['gaps']
    assert all(set(values) == {int(gap) + 1} and len(values) == 24 for gap, values in gaps.items()), gaps
    print('empty-interval calibration: C(M) = M + 1 in all', 24 * len(gaps), 'samples')
    for family, family_models in models.items():
        for key, model in family_models.items():
            print(family, key, f'{model["intercept_cycles"]:.3f} + {model["cycles_per_kib"]:.5f}K', 'discovery max median error', round(model['discovery']['max_median_error_cycles'], 2))

def formula(model: dict[str, Any]) -> str:
    return f'{model["intercept_cycles"]:.3f} + {model["cycles_per_kib"]:.5f}K'

def validate(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    models = json.loads((root / 'models.json').read_text())
    checked: dict[str, dict[str, dict[str, Any]]] = {}
    for family in ('dma', 'pull', 'host'):
        validation = groups(load(root, f'{family}_validation'), family)
        assert set(validation) == set(models[family]), family
        checked[family] = {key: {**models[family][key], 'validation': errors(configs, models[family][key]), 'validation_kib': [config['size_bytes'] // 1024 for config in configs]} for key, configs in validation.items()}
    return checked

def sort_key(key: str) -> tuple[str, int]:
    source, _, window = key.rpartition('/')
    return source, int(window)

def write_tables(root: Path, checked: dict[str, dict[str, dict[str, Any]]]) -> None:
    rows = ['| W | raw 周期中心公式，K=S/1024 | 验证中位数最大误差 | 验证全样本残差 |', '| ---: | --- | ---: | --- |']
    for key in sorted(checked['dma'], key=int):
        model = checked['dma'][key]
        check = model['validation']
        rows.append(f'| {key} | {formula(model)} | {check["max_median_error_cycles"]:.2f} | [{check["min_residual_cycles"]:.2f},{check["max_residual_cycles"]:.2f}] |')
    samples = sum(model['discovery']['samples'] + model['validation']['samples'] for model in checked['dma'].values())
    head = f'HBM→BC 私有 BMEM，libtpu 0.0.50，tpuasm 的 `BarnaCoreRuntime`；同一 BC（芯片 (0,0) 的 BC 0）的 paired LCC 包围 W 条 `dma.simple` 及其完成 sflag 的 `swait.ge`。S 为每条字节数，K=S/1024；W=1 的 S 为 4 KiB–2 MiB，W=4/16 为 1–128 KiB，HBM 源为 64 MiB 环。以下为发现组冻结的中心公式及独立尺寸残差，均保留 raw 端点开销，共 {samples} 个样本。[复现与完整条件](../README.md#5-hbm--bc-私有-bmem)。DMA 尺寸须为 512 B 的整数倍；公式针对正文列出的已测条件。libtpu 0.0.46 的旧公式见[历史对照](05_bc_libtpu_0_0_46.md)。'
    (RESULTS / '05_bc.md').write_text(head + '\n\n' + '\n'.join(rows) + '\n')
    rows = ['| 来源 | W | raw 周期中心公式，K=S/1024 | 验证中位数最大误差 | 验证全样本残差 |', '| --- | ---: | --- | ---: | --- |']
    for key in sorted(checked['pull'], key=sort_key):
        model = checked['pull'][key]
        check = model['validation']
        source, window = key.split('/')
        rows.append(f'| {source} | {window} | {formula(model)} | {check["max_median_error_cycles"]:.2f} | [{check["min_residual_cycles"]:.2f},{check["max_residual_cycles"]:.2f}] |')
    samples = sum(model['discovery']['samples'] + model['validation']['samples'] for model in checked['pull'].values())
    head = f'TC VMEM（`vmem`，芯片 (0,0) 上运行中的 Pallas kernel 持有的 scratch）或 Megacore Shared CMEM（`cmem`）→BC 私有 BMEM，由 BC 发起，libtpu 0.0.50，tpuasm 的 `BarnaCoreRuntime`；同一 BC 的 paired LCC 包围 W 条 `dma.simple` 及其完成 sflag 的 `swait.ge`；TC VMEM 来源在 BC 读完后由 BC 归还 kernel 的信号量。S 为每条字节数，K=S/1024；W=1/4/16 的 S 分别为 4 KiB–2 MiB、4–512 KiB、4–128 KiB。以下为冻结中心公式及独立尺寸残差，共 {samples} 个样本。[复现与完整条件](../README.md#7-tc-vmemmegacore-shared-cmem--bc-私有-bmem)。DMA 尺寸须为 4 KiB 的整数倍；公式针对正文列出的已测条件。libtpu 0.0.46 的旧公式见[历史对照](07_bc_pull_libtpu_0_0_46.md)。'
    (RESULTS / '07_bc_pull.md').write_text(head + '\n\n' + '\n'.join(rows) + '\n')
    empty = [config for config in load(root, 'host_discovery')['configs'] if config['size_bytes'] == 0][0]
    empty_cycles = [row['cycles'] for row in empty['records']]
    rows = ['| 调用 | W | raw 周期中心公式，K=S/1024 | 验证中位数最大误差 | 最大相对误差 | 验证全样本残差 |', '| --- | ---: | --- | ---: | ---: | --- |']
    for key in sorted(checked['host']):
        model = checked['host'][key]
        check = model['validation']
        mode, window = key.split('/')
        rows.append(f'| {mode} | {window} | {formula(model)} | {check["max_median_error_cycles"]:.2f} | {100 * check["max_relative_median_error"]:.2f}% | [{check["min_residual_cycles"]:.2f},{check["max_residual_cycles"]:.2f}] |')
    head = f'HBM→host，host 发起，libtpu 0.0.50：`api` 调用 tpuasm 的 `BarnaCoreRuntime.read_hbm`（含它分配缓冲区并复制成 `bytes` 的 Python 开销），`native` 把预先分配的缓冲区直接交给同一个原生桥接调用（`TfrtDriverWrapper::ReadHbm`，即同步的 `TpuSharedMemory::Read`）。同一 BC 上的常驻计数程序以 paired LCC 包围两次 host↔BC 握手与其间的一次读取，数据在 END 之后核对。S 为 4 KiB–4 MiB，源为 64 MiB 的 HBM 环。只握手、不读取的空窗口 24 个样本中位数 {np.median(empty_cycles):.0f}、范围 [{min(empty_cycles)}, {max(empty_cycles)}]，不从公式中扣除。这不是第 9 节的 Host Magic Queue 路径，[说明](../README.md#92-libtpu-0050-上的-host-读取)。'
    (RESULTS / '09_host_read.md').write_text(head + '\n\n' + '\n'.join(rows) + '\n')

def value(words: list[int], index: int) -> int:
    return words[index] | words[index + 1] << 32

def bc_interval(entry: dict[str, Any], sample: dict[str, Any]) -> tuple[int, int, int]:
    """返回 (ΔGTC, 两个 GTC 采样点之间的 ΔLCC, 计时的那个 LCC 区间的 ΔLCC)。端点在相邻的 bundle 中，GTC 在外时 GTC 采样点之间比 LCC 区间多 2 个周期，LCC 在外时少 2 个。"""
    words = sample['words']
    outer = value(words, 6) - value(words, 0)
    inner = value(words, 4) - value(words, 2)
    if entry['outer'] == 'gtc':
        return outer, inner + 2, inner
    return inner, outer - 2, outer

def tc_interval(entry: dict[str, Any], sample: dict[str, Any]) -> tuple[int, int, int]:
    """TC 内联块的 8 个输出：外层 BEGIN、内层 BEGIN、内层 END、外层 END 的低 32 位，再是同样次序的高 32 位。"""
    words = sample['words']
    point = [words[i] | words[i + 4] << 32 for i in range(4)]
    outer, inner = point[3] - point[0], point[2] - point[1]
    if entry['outer'] == 'gtc':
        return outer, inner + 2, inner
    return inner, outer - 2, outer

def ppm(gtc: list[int], lcc: list[int]) -> float:
    slope = np.polyfit(np.array(lcc, dtype=float), np.array(gtc, dtype=float), 1)[0]
    return float((slope / TICKS_PER_CYCLE - 1) * 1e6)

def clock_tables(root: Path) -> None:
    document = load(root, 'clock')
    bc_rows = ['| core_index | 芯片 | BC core | 外层 | 区间数 | spin ppm | DMA ppm | host_wait ppm | 全部 ppm | 残差 ticks | 最长 ΔLCC | host ns／(ΔGTC/11.2) |', '| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    per_chip: dict[str, dict[str, list[float]]] = {}
    increments: list[int] = []
    windows: list[int] = []
    lcc_steps: list[int] = []
    intervals = 0
    for entry in document['bc']:
        samples = [s for s in entry['samples'] if s['pass'] != 'warmup']
        assert all(s['finished'] for s in entry['samples']), entry['core_index']
        assert all(s.get('payload_and_guards_correct', True) for s in entry['samples']), entry['core_index']
        timed = [s for s in samples if s['mode'] in ('spin', 'dma', 'host_wait')]
        pairs = [(s, bc_interval(entry, s)) for s in timed]
        points = {mode: [p for s, p in pairs if s['mode'] == mode] for mode in ('spin', 'dma', 'host_wait')}
        for s in timed:
            assert s['mode'] == 'host_wait' or s['words'][8] == s['count'], (entry['core_index'], s['mode'], s['words'][8], s['count'])
        for s, (_, _, measured) in zip([s for s in timed if s['mode'] == 'spin'], points['spin']):
            assert measured == 4 * s['count'] + (1 if entry['outer'] == 'gtc' else 3), (entry['core_index'], entry['outer'], s['count'], measured)
        for s in samples:
            if s['mode'].startswith('dense'):
                reads = [value(s['words'], 2 * i) for i in range(6)]
                steps = [b - a for a, b in zip(reads, reads[1:])]
                if s['mode'] == 'dense_lcc':
                    lcc_steps.extend(steps)
                else:
                    increments.extend(steps)
                    windows.extend(sum(steps[i:i + 3]) for i in range(len(steps) - 2))
        everything = [p for _, p in pairs]
        intervals += len(everything)
        gtc, lcc = [p[0] for p in everything], [p[1] for p in everything]
        residual = [g - TICKS_PER_CYCLE * l for g, l in zip(gtc, lcc)]
        host_x = [p[0] / GTC_PER_NS for p in everything]
        host_y = [s['host_ns'] for s, _ in pairs]
        host_slope = float(np.polyfit(np.array(host_x), np.array(host_y, dtype=float), 1)[0])
        chip = '(' + ','.join(str(c) for c in entry['chip'][:2]) + ')'
        overall = ppm(gtc, lcc)
        per_chip.setdefault(chip, {'bc': [], 'short': [], 'detrended': [], 'points': []})['bc'].append(overall)
        per_chip[chip]['points'].extend(everything)
        slope = np.polyfit(np.array(lcc, dtype=float), np.array(gtc, dtype=float), 1)
        per_chip[chip]['short'].extend(r for r, l in zip(residual, lcc) if l < 10000)
        per_chip[chip]['detrended'].extend((g - np.polyval(slope, l)) for g, l in zip(gtc, lcc))
        cells = [f'{ppm([p[0] for p in points[mode]], [p[1] for p in points[mode]]):+.2f}' for mode in ('spin', 'dma', 'host_wait')]
        bc_rows.append(f'| {entry["core_index"]} | {chip} | {entry["core"]} | {entry["outer"].upper()} | {len(everything)} | {" | ".join(cells)} | {overall:+.2f} | [{min(residual):.1f}, {max(residual):.1f}] | {max(lcc):,} | {host_slope:.6f} |')
    tc_rows = ['| JAX 芯片 | 芯片 | 区间数 | ΔLCC−H | 全部 ppm | 残差 ticks | 最长 ΔLCC |', '| --- | --- | ---: | --- | ---: | --- | ---: |']
    chips: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for entry in document['tc']:
        chip = '(' + ','.join(str(c) for c in entry['chip'][:2]) + ')'
        chips.setdefault(chip, []).extend((entry, s) for s in entry['samples'][1:])
    for chip, items in chips.items():
        points = [tc_interval(entry, s) for entry, s in items]
        fixed = sorted({(measured - s['delay'] - (2 if entry['outer'] == 'lcc' else 0)) for (entry, s), (_, _, measured) in zip(items, points)})
        gtc, lcc = [p[0] for p in points], [p[1] for p in points]
        residual = [g - TICKS_PER_CYCLE * l for g, l in zip(gtc, lcc)]
        overall = ppm(gtc, lcc)
        per_chip[chip]['tc'] = [overall]
        tc_rows.append(f'| {chip} | {chip} | {len(points)} | {", ".join(str(v) for v in fixed)} | {overall:+.2f} | [{min(residual):.1f}, {max(residual):.1f}] | {max(lcc):,} |')
    merged = ['| 芯片 | BC ppm（合并） | 8 组的范围 | TC0 ppm | 短区间残差（ΔLCC<10⁴） | 扣除斜率后的残差 |', '| --- | ---: | --- | ---: | --- | --- |']
    for chip in ('(0,0)', '(1,0)', '(0,1)', '(1,1)'):
        info = per_chip[chip]
        together = ppm([p[0] for p in info['points']], [p[1] for p in info['points']])
        merged.append(f'| {chip} | {together:+.3f} | {min(info["bc"]):+.3f} 至 {max(info["bc"]):+.3f} | {info["tc"][0]:+.3f} | [{min(info["short"]):.0f}, {max(info["short"]):.0f}] | [{min(info["detrended"]):.0f}, {max(info["detrended"]):.0f}] |')
    good = sum(step in (1, 15, 16) for step in increments)
    head = f'BC 本地 LCC 与 GTC 的计数比例，libtpu 0.0.50，tpuasm 的 `BarnaCoreRuntime`。每个 BC 运行两种嵌套顺序（GTC 在外／LCC 在外），区间内依次为空循环 spin、N 次 HBM→BC 私有 BMEM DMA（每次复位完成 sflag、发起、`swait.ge`），以及 `swait.gt` 等待 host 写 sflag 的 host_wait；共 {intervals} 个区间。ppm 列为 ΔGTC 对 ΔLCC 的拟合斜率相对 TC 的 32/3 的偏差；ΔLCC 已按内外层相差一个 bundle 的端点换算到两个 GTC 采样点之间。残差为每个样本的 ΔGTC−(32/3)ΔLCC，单位为 raw GTC tick。host 列为 host 从 `start()` 到看到完成标记的时间对 ΔGTC/11.2 的拟合斜率，只作 GTC 时基的粗核对。连续读取：paired LCC 的 {len(lcc_steps)} 个相邻增量全部为 {sorted(set(lcc_steps))}；paired GTC 的 {len(increments)} 个相邻增量中 {good} 个属于 1、15、16（其余：{sorted(step for step in increments if step not in (1, 15, 16))}），{len(windows)} 个三周期窗口中 {sum(w == 32 for w in windows)} 个合计 32 ticks。[方法与结论](../README.md#12-bc-本地周期与-tc-周期)。libtpu 0.0.46 的旧表见[历史对照](15_bc_clock_libtpu_0_0_46.md)。'
    tail = '同芯片对照：每颗芯片的 TC0 在 Pallas kernel 的 tpuasm 内联块中嵌套读取 LCC 与 GTC，中间以 `vdelay` 加 `sfence` 拉长区间，两种嵌套顺序各测；ΔLCC−H 列为扣除 vdelay 长度 H 后的固定周期（LCC 在外时另扣两端各一个 bundle）。'
    (RESULTS / '15_bc_clock.md').write_text(head + '\n\n' + '\n'.join(bc_rows) + '\n\n' + tail + '\n\n' + '\n'.join(tc_rows) + '\n\n按芯片合并：BC ppm（合并）是该芯片 4 个 BC × 2 种顺序全部区间一起拟合的斜率，另列 8 组各自的范围；残差单位为 raw GTC tick。\n\n' + '\n'.join(merged) + '\n')
    print('clock tables written:', intervals, 'BC intervals')

def main(arguments: argparse.Namespace) -> None:
    root = arguments.output
    if arguments.freeze:
        freeze(root)
        return
    checked = validate(root)
    for family, family_models in checked.items():
        for key, model in family_models.items():
            check = model['validation']
            print(family, key, formula(model), 'validation max median error', round(check['max_median_error_cycles'], 2), f'({100 * check["max_relative_median_error"]:.2f}%)', 'residuals', [round(check['min_residual_cycles'], 2), round(check['max_residual_cycles'], 2)])
    write_tables(root, checked)
    if (root / 'clock.json').exists():
        clock_tables(root)
