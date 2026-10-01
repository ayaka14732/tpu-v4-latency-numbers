"""顺序校准远程探针的四寄存器与 SMEM 端点，真实 DMA 和 credit 位于校准区间外。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

def main(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(here.parents[1] / 'tpuasm' / 'src')
    intervals = 0
    for window in (1, 8):
        for gap in (0, 1, 4, 16, 64):
            name = f'w{window}_gap{gap}'
            if args.cases and name not in args.cases.split(','):
                continue
            output = args.output / name
            summary = output / 'summary.json'
            if not summary.exists():
                assert not args.verify_only, f'缺少已完成记录：{summary}'
                command = [sys.executable, str(here / '04_remote_dma.py'), '--space', 'cmem', '--chips', '4', '--window', str(window), '--gap', str(gap)]
                command += ['--size', '262144', '--sizes', '4096', '--output', str(output)]
                print('starting', name, flush=True)
                with (args.output / f'{name}.log').open('w') as stream:
                    subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            data = json.loads(summary.read_text())
            assert data['protocol'] == 'remote_lcc_v3_all_credits_first' and data['baseline_after'] and data['repeats'] == 24
            record, = data['records']
            assert record['correct'] and record['window'] == window and record['size_bytes'] == 4096 and record['capacity_bytes'] == 262144
            assert record['audit']['gap_bundles'] == gap
            assert record['audit']['counter_storage'] == ('registers' if window == 1 else 'smem')
            counts = np.asarray(record['lcc'], dtype=np.uint64)
            assert counts.shape == (24, 4, 2)
            np.testing.assert_array_equal(counts[:, :, 1] - counts[:, :, 0], record['cycles'])
            np.testing.assert_array_equal(record['cycles'], np.full((24, 4), gap + (12 if window == 1 else 21)))
            intervals += 96
            print('verified', name, flush=True)
    print(intervals, 'TC calibration intervals verified')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/04_remote_controls'))
    parser.add_argument('--cases', help='逗号分隔的校准名，例如 w8_gap0')
    parser.add_argument('--verify-only', action='store_true', help='只用 CPU 核对已有记录')
    main(parser.parse_args())
