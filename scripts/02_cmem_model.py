"""检查单 TC CMEM ↔ TC VMEM 的 payload/请求发射分段公式，不拟合常数。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np

local = import_module('02_local_model')

def prediction(path: str, size: int, window: int) -> float:
    if path == 'vmem_cmem':
        return (309 if window == 1 else 310) + window * size / 1024
    if path == 'cmem_vmem':
        return 311 + size / 2048 if window == 1 else 310 + window * max(3, size / 2048)
    raise ValueError(path)

def verify(root: Path, phase: str) -> dict:
    records = local.load_phase(root, phase, ready_only=phase == 'discovery')
    groups = {}
    for record in records:
        if record['cores'] != 1 or record['size_bytes'] < 4096 or record['path'] not in ('cmem_vmem', 'vmem_cmem'):
            continue
        key = local.model_key(record)
        estimate = prediction(record['path'], record['size_bytes'], record['window'])
        groups.setdefault(key, []).extend(float(value - estimate) for value in record['cycles'])
    summary = {}
    for key, values in groups.items():
        data = np.asarray(values)
        summary[key] = {'samples': len(data), 'min_residual': float(data.min()), 'max_residual': float(data.max()), 'within_one_cycle': int(np.sum(np.abs(data) <= 1))}
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/tmp/tpu_latency_numbers/02_local_suite'))
    parser.add_argument('--phase', choices=('discovery', 'validation'), default='discovery')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.phase), indent=2))
