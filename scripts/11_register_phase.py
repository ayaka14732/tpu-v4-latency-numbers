"""冻结 BEGIN 奇偶位候选，再用后续正式配置验证；不改写原始固定候选或每组冻结模型。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from importlib import import_module
import json
from pathlib import Path

model = import_module('11_register_model')
suite = import_module('11_register_suite')
CALIBRATION_CASES = {
    'base_s64_u1_a1_c1',
    'base_s64_u1_a1_c2',
    'base_s4096_u32_a32_c2',
    'a8_s1024_u128_a8_c2',
    'cmem_a2_s1024_u128_a2_c2',
}

def specification() -> dict:
    return dict(
        version='register_begin_parity_v1',
        vmem='C=(2U+5+4I)N+13',
        cmem='C=(54ceil(U/16)+2min(U,16)+6I)N+12+I+(BEGIN mod 2)',
        spill='I=1 iff U=A=32, otherwise I=0',
        calibration_cases=sorted(CALIBRATION_CASES),
        independent_cases='the other 47 formal matrix configurations, both discovery and validation runs',
    )

def freeze(root: Path) -> None:
    path = root / 'phase_candidate.json'
    if path.exists():
        assert json.loads(path.read_text()) == specification(), '不能修改已冻结的奇偶位候选'
        return
    actual = {path.parent.name.rsplit('_', 1)[0] for path in root.glob('*/summary.json')}
    assert actual == CALIBRATION_CASES, '此候选只能在其余 47 个正式配置采样之前冻结'
    assert {path.name.removesuffix('.validated.json') for path in root.glob('*.validated.json')} == CALIBRATION_CASES
    path.write_text(json.dumps(specification(), indent=2) + '\n')

def verify(root: Path, output: Path, partial: bool) -> None:
    assert json.loads((root / 'phase_candidate.json').read_text()) == specification()
    expected = {f'{job["family"]}_s{job["size_kib"]}_u{job["streams"]}_a{job["accumulators"]}_c{job["cores"]}' for job in suite.jobs()}
    actual = {path.name.removesuffix('.validated.json') for path in root.glob('*.validated.json')}
    assert actual <= expected and CALIBRATION_CASES <= actual
    assert partial or actual == expected, f'尚缺 {len(expected - actual)} 个正式配置'
    residuals = defaultdict(list)
    for name in sorted(actual):
        category = '构建候选的五组' if name in CALIBRATION_CASES else '冻结后新增配置'
        for phase in ('discovery', 'validation'):
            path = root / f'{name}_{phase}' / 'summary.json'
            suite.records(path)
            data = json.loads(path.read_text())
            assert data['libtpu'] == '0.0.49'
            for row in data['records']:
                slope, intercept = model.candidate(row['space'], row['streams'], row['accumulators'])
                for sample in row['records']:
                    for lcc, cycles in zip(sample['lcc'], sample['cycles'], strict=True):
                        parity = lcc[0] & 1
                        predicted = slope * row['iterations'] + intercept + (parity if row['space'] == 'cmem' else 0)
                        residuals[(category, row['space'], parity)].append(cycles - predicted)
    rows = ['| 数据用途 | 内存 | BEGIN mod 2 | TC 区间数 | 最小残差 | 最大残差 | 非零残差数 |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    total, mismatches = 0, 0
    for (category, space, parity), values in sorted(residuals.items()):
        failed = sum(value != 0 for value in values)
        rows.append(f'| {category} | {space} | {parity} | {len(values)} | {min(values)} | {max(values)} | {failed} |')
        total += len(values)
        mismatches += failed
    header = (
        'BEGIN 奇偶位候选：TC VMEM 为 `(2U+5+4I)N+13`；Megacore Shared CMEM 为 '
        '`(54⌈U/16⌉+2min(U,16)+6I)N+12+I+(BEGIN mod 2)`，其中 U=A=32 时 I=1，其余 I=0。'
        'N 为循环次数；工作集、累加器与端点条件见[循环读取](../README.md#11-保留工作集与累加器条件的循环读取)。\n\n'
        '候选在首批五个正式配置完成后、其余 47 个正式配置开始前冻结。首批五组只用于构建候选，'
        '后续配置的发现与验证采样都作为此候选的新数据；它们原有的逐配置冻结模型另行保留。'
        '下表按 BEGIN 的最低位分别报告原始整数残差，不由后续样本调整系数，也不据此指定硬件时钟域的原因。\n\n'
    )
    header += f'已核对 {len(actual)}/52 个配置、{total} 个 TC 区间，其中冻结后新增 {len(actual - CALIBRATION_CASES)}/47 个配置；非零残差 {mismatches} 个。\n\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + '\n'.join(rows) + '\n')
    print(len(actual), '/52 configurations;', total, 'TC intervals;', mismatches, 'parity-candidate mismatches;', output)
    assert mismatches == 0, 'BEGIN 奇偶位候选出现反例；保留表中残差，不修改冻结公式'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/tmp/tpu_latency_numbers/11_register_suite'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/11_register_phase.md'))
    parser.add_argument('--freeze', action='store_true', help='只在其余配置采样之前冻结此候选，并检查现有五组')
    parser.add_argument('--partial', action='store_true', help='允许尚未完成全部配置；不代表完整独立验证')
    args = parser.parse_args()
    if args.freeze:
        freeze(args.root)
    verify(args.root, args.output, args.partial or args.freeze)
