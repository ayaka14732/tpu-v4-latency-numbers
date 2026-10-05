"""在 CPU 上核对 CLD/CRF 服务间隔探针的逐样本整数公式。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

def expected_gaps(name: str) -> tuple[int, int]:
    if match := re.fullmatch(r'load-(dense|vnop)-n(\d+)', name):
        spacing, count_text = match.groups()
        count = int(count_text)
        if spacing == 'dense':
            return count + 1, max(count + 13, 2 * count + 11)
        return 2 * count, 2 * count + 12
    if match := re.fullmatch(r'pop-vr[01]-(?:same|round-robin)-n(\d+)', name):
        count = int(match[1])
        return count + 1, max(count + 13, 2 * count + 11)
    if match := re.fullmatch(r'pop-erf-n(\d+)', name):
        count = int(match[1])
        return count + 1, count + 13
    if name in ('pop-use-independent', 'pop-use-dependent'):
        return 3, 15
    raise ValueError(name)

def verify(paths: list[Path]) -> dict:
    configurations, samples = 0, 0
    for path in paths:
        data = json.loads(path.read_text())
        assert data['baseline_after'], path
        for record in data['records']:
            expected = expected_gaps(record['name'])
            assert all(record['correct']), (path, record['name'])
            assert all(value == 1024 for value in record['matching_words']), (path, record['name'])
            for lcc, gaps in zip(record['lcc'], record['gaps_from_begin'], strict=True):
                assert tuple(gaps) == expected, (path, record['name'], gaps, expected)
                assert [lcc[1] - lcc[0], lcc[2] - lcc[0]] == gaps
                samples += 1
            configurations += 1
    return {'configurations': configurations, 'samples': samples, 'all_formula_matches': True}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('/tmp/tpu_latency_numbers/03_cld_service')
    parser.add_argument('inputs', nargs='*', type=Path, default=[root / 'discovery' / 'summary.json', root / 'holdout' / 'summary.json'])
    args = parser.parse_args()
    if not args.inputs:
        parser.error('需要至少一个已采样的 summary JSON')
    print(json.dumps(verify(args.inputs), indent=2))
