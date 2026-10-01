"""在 CPU 上核对第 3 节 CLD 探针的逐样本整数公式。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

def cycles(name: str) -> int:
    if match := re.fullmatch(r'(?:stream|fused)-n(\d+)-d(\d+)', name):
        count, depth = map(int, match.groups())
        if depth == 0:
            return 56 * count + 11
        if not 1 <= depth <= 28:
            raise ValueError(f'D={depth} 超出本探针的预取范围')
        # D=27/28 推翻旧的常数 67：预取和排空本身开始超过 CRF 等待。
        return max(67, 2 * depth + 14) + max(54, 2 * depth) * ((count - 1) // depth) + 2 * ((count - 1) % depth)
    if match := re.fullmatch(r'pop-use-(\d+)', name):
        return 67 + int(match[1])
    if match := re.fullmatch(r'cld-pop-(\d+)', name):
        return max(67, int(match[1]) + 14)
    raise ValueError(name)

def verify(paths: list[Path]) -> dict:
    configurations, samples = 0, 0
    for path in paths:
        data = json.loads(path.read_text())
        assert data['baseline_after'], path
        for record in data['records']:
            expected = cycles(record['name'])
            assert all(record['correct']), (path, record['name'])
            assert all(value == 1024 for value in record['matching_words']), (path, record['name'])
            for lcc, gaps in zip(record['lcc'], record['gaps_from_begin'], strict=True):
                assert [lcc[1] - lcc[0], lcc[2] - lcc[0]] == gaps
                assert gaps[1] == expected, (path, record['name'], gaps, expected)
                samples += 1
            configurations += 1
    return {'configurations': configurations, 'samples': samples, 'all_formula_matches': True}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('/tmp/tpu_latency_numbers/evidence/03_cld')
    parser.add_argument('inputs', nargs='*', type=Path, default=[root / f'{name}.json' for name in ('pilot', 'pipeline', 'holdout', 'deep', 'deep_holdout', 'fused', 'fused_holdout')])
    args = parser.parse_args()
    if not args.inputs:
        parser.error('需要至少一个已采样的 summary JSON')
    print(json.dumps(verify(args.inputs), indent=2))
