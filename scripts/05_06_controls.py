"""复现 BC paired LCC、Host SMEM 计数器与 issue-only 对照；串行使用两个 runtime。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

model = import_module('05_06_dma_model')

def jobs() -> list[dict]:
    result = []
    for family, window, offset in (('05_bc', 1, 1), ('06_host', 16, 21)):
        for gap in (0, 1, 4, 16, 64):
            result.append(dict(family=family, name=f'gap{gap}', size=4096, window=window, extra=['--gap', str(gap)], cycles=gap + offset))
    for size in (4096, 1048576):
        result.append(dict(family='06_host', name=f'issue_s{size}', size=size, window=1, extra=['--boundary', 'issue_only'], cycles=2))
    return result

def main(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    total = 0
    for job in jobs():
        output = args.output / f'{job["family"]}_{job["name"]}'
        summary = output / 'summary.json'
        if not summary.exists():
            assert not args.archive_only, f'缺少已完成记录：{summary}'
            output.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(args.libtpu_root) if job['family'] == '05_bc' else str(here.parents[1] / 'tpuasm' / 'src')
            command = [sys.executable, str(here / f'{job["family"]}_dma.py'), '--size', str(job['size']), '--window', str(job['window']), '--output', str(output), *job['extra']]
            print('starting', output.name, flush=True)
            with (output / 'run.log').open('w') as stream:
                subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            time.sleep(1)
        data = json.loads(summary.read_text())
        assert data['size_bytes'] == job['size'] and data['window'] == job['window']
        audit = data if job['family'] == '05_bc' else data['audit']
        if job['name'].startswith('gap'):
            assert audit['gap_bundles'] == int(job['name'][3:])
        else:
            assert audit['boundary'] == 'issue_only'
        record = model.records(summary)
        assert len(record['cycles']) == 24 and set(record['cycles']) == {job['cycles']}
        total += len(record['cycles'])
        destination = args.archive / job['family']
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(summary, destination / f'{job["name"]}.json')
        if job['family'] == '06_host' and job['name'] in ('gap64', 'issue_s1048576'):
            name = 'gap64.tpuasm' if job['name'] == 'gap64' else 'issue_1m.tpuasm'
            shutil.copyfile(output / 'instrumented.tpuasm', destination / name)
    print(total, 'calibration/control samples verified and archived')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers'))
    parser.add_argument('--archive', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence'))
    parser.add_argument('--libtpu-root', type=Path, default=Path('/tmp/libtpu-0.0.46'))
    parser.add_argument('--archive-only', action='store_true', help='仅核对和整理已有记录，不启动 TPU 进程')
    main(parser.parse_args())
