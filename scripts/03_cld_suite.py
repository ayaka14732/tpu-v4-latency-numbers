"""复现第 3 节全部七组 CLD 实验，校验配置后将原始记录归档到仓库外。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

model = import_module('03_cld_model')

def jobs() -> list[dict]:
    return [
        dict(name='pilot', group='all', counts=[1, 4, 16, 64, 128], depths=[0, 1], repeats=12, schedule='split'),
        dict(name='pipeline', group='streams', counts=[4, 16, 64, 128], depths=[2, 4], repeats=24, schedule='split'),
        dict(name='holdout', group='streams', counts=[3, 7, 31, 65], depths=[0, 1, 2, 4], repeats=24, schedule='split'),
        dict(name='deep', group='streams', counts=[8, 16, 31, 64, 128, 256], depths=[8, 16, 24, 27, 28], repeats=24, schedule='split'),
        dict(name='deep_holdout', group='streams', counts=[29, 55, 97], depths=[8, 16, 24, 26, 27, 28], repeats=24, schedule='split'),
        dict(name='fused', group='streams', counts=[4, 16, 32, 64, 128], depths=[1, 2, 4, 8, 16, 24, 26, 27, 28], repeats=24, schedule='fused'),
        dict(name='fused_holdout', group='streams', counts=[29, 55, 97], depths=[1, 2, 4, 8, 16, 24, 26, 27, 28], repeats=24, schedule='fused'),
    ]

def verify_job(path: Path, job: dict) -> None:
    data = json.loads(path.read_text())
    prefix = 'stream' if job['schedule'] == 'split' else 'fused'
    expected = {f'{prefix}-n{count}-d{depth}' for count in job['counts'] for depth in job['depths'] if depth <= count}
    if job['group'] == 'all':
        expected |= {f'pop-use-{distance}' for distance in (0, 1, 2, 4, 8, 16, 32, 64)}
        expected |= {f'cld-pop-{distance}' for distance in (1, 2, 4, 8, 16, 32, 64, 128, 256)}
    assert {record['name'] for record in data['records']} == expected
    assert data['environment']['repeats'] == job['repeats']
    assert all(len(record['lcc']) == job['repeats'] for record in data['records'])
    model.verify([path])

def main(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    args.archive.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(here.parents[1] / 'tpuasm' / 'src')
    summaries, counterexamples = [], []
    for job in jobs():
        output = args.output / f'03_cld_{job["name"]}'
        summary = output / 'summary.json'
        if not summary.exists():
            assert not args.archive_only, f'缺少已完成记录：{summary}'
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(here / '03_cld.py'), '--output', str(output)]
            for key in ('group', 'counts', 'depths', 'repeats', 'schedule'):
                value = ','.join(map(str, job[key])) if isinstance(job[key], list) else str(job[key])
                command += ['--' + key, value]
            print('starting', job['name'], flush=True)
            with (output / 'run.log').open('w') as stream:
                subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        verify_job(summary, job)
        destination = args.archive / f'{job["name"]}.json'
        shutil.copyfile(summary, destination)
        summaries.append(destination)
        for path in output.glob('*.full.tpuasm'):
            shutil.copyfile(path, args.archive / path.name)
        if job['name'] == 'deep':
            for record in json.loads(summary.read_text())['records']:
                match = re.fullmatch(r'stream-n(\d+)-d(\d+)', record['name'])
                assert match is not None
                count, depth = map(int, match.groups())
                prediction = 67 + max(54, 2 * depth) * ((count - 1) // depth) + 2 * ((count - 1) % depth)
                actual = sorted({gaps[1] for gaps in record['gaps_from_begin']})
                if actual != [prediction]:
                    counterexamples.append(dict(name=record['name'], predicted_cycles=prediction, actual_cycles=actual))
    rejected = dict(formula='67 + max(54,2D)*floor((N-1)/D) + 2*((N-1)%D)', counterexamples=counterexamples)
    (args.archive / 'rejected_deep_model.json').write_text(json.dumps(rejected, indent=2) + '\n')
    print(model.verify(summaries))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers'))
    parser.add_argument('--archive', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/03_cld'))
    parser.add_argument('--archive-only', action='store_true', help='仅核对和整理已有记录，不启动 TPU 进程')
    main(parser.parse_args())
