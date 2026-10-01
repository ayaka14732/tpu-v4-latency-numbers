"""用已有双 TC 环形 HBM 记录说明大小中心模型的限制，不修改冻结公式。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from importlib import import_module
import json
from pathlib import Path

import numpy as np

ring = import_module('10_hbm_ring_suite')

def analyze(root: Path, output: Path) -> None:
    distributions = ['| 路径 | 数据组 | S（KiB） | TC0 最小／中位数／最大 | TC1 最小／中位数／最大 | 成对相关系数 |', '| --- | --- | ---: | --- | --- | ---: |']
    repeats = ['| 路径 | S（KiB） | 逻辑环槽 | 两次调用序号 | 第一次 (C₀,C₁) | 第二次 (C₀,C₁) |', '| --- | ---: | ---: | --- | --- | --- |']
    samples = 0
    for phase in ('discovery', 'validation'):
        path = root / f'c2_w1_{phase}' / 'summary.json'
        ring.records(path)
        document = json.loads(path.read_text())
        for row in document['records']:
            if row['path'] not in ('hbm_cmem', 'cmem_hbm') or row['size_bytes'] < 1048576:
                continue
            assert row['cores'] == 2 and row['window'] == 1
            cycles = np.asarray([sample['cycles'] for sample in row['records']], dtype=np.int64)
            assert cycles.shape == (24, 2)
            limits = [np.quantile(cycles[:, core], (0, 0.5, 1)) for core in range(2)]
            rendered = [' / '.join(f'{value:g}' for value in values) for values in limits]
            correlation = float(np.corrcoef(cycles.T)[0, 1])
            name, kib = row['path'], row['size_bytes'] // 1024
            distributions.append(f'| {name} | {phase} | {kib} | {rendered[0]} | {rendered[1]} | {correlation:.4f} |')
            slots = defaultdict(list)
            for index, sample in enumerate(row['records']):
                slots[sample['source_slot']].append((index, sample['cycles']))
            for slot, observations in sorted(slots.items()):
                for (first, a), (second, b) in zip(observations, observations[1:]):
                    repeats.append(f'| {name} | {kib} | {slot} | {first} → {second} | ({a[0]}, {a[1]}) | ({b[0]}, {b[1]}) |')
            samples += len(cycles)
    assert samples == 144
    note = (
        '双 TC、W=1、每 TC 64 MiB HBM 环，S=1 MiB、4 MiB 来自发现组，S=2 MiB 来自独立验证组；'
        '共 144 次调用、288 个本地周期区间。`cmem` 指 Megacore Shared CMEM。'
        'C₀、C₁ 各自是同一 TC 的 END−BEGIN，没有相减不同 TC 的计数器。相关系数只描述同次调用的两项局部持续周期。\n\n'
        '下表是原始样本的回看分析，原冻结公式和残差不变。重复逻辑环槽保留相同的窗口偏移，'
        '但每次重新准备 poison，HBM 目标分配及前序服务状态未单独控制；因此不能由相关性指定唯一的硬件原因。'
        '正文的逐路径公式是统计中心，这些样本展示其单次预测限制。\n\n'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(note + '\n'.join(distributions) + '\n\n调用序号从 0 开始，仅比较同一尺寸配置内的重复逻辑环槽：\n\n' + '\n'.join(repeats) + '\n')
    print(samples, 'paired calls checked;', len(repeats) - 2, 'repeated-slot comparisons;', output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/tmp/tpu_latency_numbers/10_hbm_ring_suite'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/10_hbm_variability.md'))
    args = parser.parse_args()
    analyze(args.root, args.output)
