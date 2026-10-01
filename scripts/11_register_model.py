"""以固定指令结构预测循环读取周期；保留超出候选公式的样本，不由验证数据修改系数。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np

suite = import_module('11_register_suite')

def candidate(space: str, streams: int, accumulators: int) -> tuple[int, int]:
    assert space in ('vmem', 'cmem') and streams in (1, 4, 8, 16, 32, 64, 128)
    assert 1 <= accumulators <= min(streams, 32) and (accumulators != 32 or streams == 32)
    spill = accumulators == 32
    if space == 'vmem':
        # 每轮 U 次 load/add、地址推进、回卷、分支及 delay bundle；A=32 多四条 spill 指令。
        return 2 * streams + 5 + 4 * spill, 13
    # D<=16 的首组等待与 pop/add 排空；后续 CRF 组按同样的 54-cycle 距离推进。
    # CMEM 流插入两条 vld 时，还改变向量发射相对标量的距离，候选额外成本为六周期。
    return 54 * ((streams + 15) // 16) + 2 * min(streams, 16) + 6 * spill, 12 + spill

def verify(paths: list[Path], output: Path) -> None:
    rows = ['| 配置 | 内存 | TC | 候选周期公式 | 最小残差 | 最大残差 | 样本数 |', '| --- | --- | ---: | --- | ---: | ---: | ---: |']
    total, mismatches = 0, 0
    for path in paths:
        document = json.loads(path.read_text())
        suite.records(path)
        groups = {}
        for record in document['records']:
            slope, intercept = candidate(record['space'], record['streams'], record['accumulators'])
            prediction = slope * record['iterations'] + intercept
            for core in range(record['cores']):
                key = (record['space'], core, slope, intercept)
                values = np.array([row['cycles'][core] for row in record['records']]) - prediction
                groups.setdefault(key, []).extend(values.tolist())
                allowed = (values == 0) if record['space'] == 'vmem' or record['cores'] == 1 else ((values == 0) | (values == 1))
                mismatches += int(np.count_nonzero(~allowed))
                total += len(values)
        for (space, core, slope, intercept), values in groups.items():
            rows.append(f'| {path.parent.name} | {space} | {core} | {slope}N+{intercept} | {min(values)} | {max(values)} | {len(values)} |')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(rows) + '\n')
    print(total, 'intervals;', mismatches, 'outside fixed candidate bounds; table:', output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='*', type=Path)
    parser.add_argument('--root', type=Path, default=Path('/tmp/tpu_latency_numbers/11_register_suite'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/11_register_candidates.md'))
    args = parser.parse_args()
    paths = args.inputs or sorted(args.root.glob('*/summary.json'))
    if not paths:
        parser.error('需要已完成的循环读取采样；没有数据不能判定候选公式成立')
    verify(paths, args.output)
