"""顺序执行本地 DMA 的发现、独立尺寸验证与端点负对照；产物默认留在 /tmp。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def jobs() -> list[dict]:
    result = []
    for phase, seed in (('discovery', 410), ('validation', 411)):
        for cores in (1, 2):
            for window in (1, 4, 8):
                if phase == 'discovery':
                    sizes = [512, 1024, 1536, 2048, 2560, 3072, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304] if window == 1 else [4096, 16384, 65536, 262144]
                else:
                    sizes = [6144, 12288, 24576, 49152, 98304, 196608, 393216, 786432, 1572864, 3145728] if window == 1 else [8192, 32768, 131072]
                result.append(dict(name=f'{phase}_c{cores}_w{window}', phase=phase, seed=seed, cores=cores, window=window, sizes=sizes, boundary='complete'))
    result.append(dict(name='issue_only', phase='control', seed=412, cores=1, window=1, sizes=[4096, 65536, 1048576], boundary='issue_only'))
    return result

def main(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(here.parents[1] / 'tpuasm' / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    matrix = jobs()
    (args.output / 'matrix.json').write_text(json.dumps(matrix, indent=2) + '\n')
    for job in matrix:
        if args.phase != 'all' and args.phase != job['phase']:
            continue
        output = args.output / job['name']
        status_file = args.output / f'{job["name"]}.status.json'
        if status_file.exists() and json.loads(status_file.read_text()).get('exit_code') == 0:
            print('already complete', job['name'], flush=True)
            continue
        command = [sys.executable, str(here / '02_local_dma.py'), '--output', str(output), '--sizes', ','.join(map(str, job['sizes']))]
        for key in ('cores', 'window', 'seed', 'boundary'):
            command += ['--' + key, str(job[key])]
        command += ['--repeats', str(args.repeats)]
        status = {'job': job, 'command': command, 'exit_code': None}
        status_file.write_text(json.dumps(status, indent=2) + '\n')
        print('starting', job['name'], flush=True)
        with (args.output / f'{job["name"]}.log').open('w') as stream:
            process = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        status['exit_code'] = process.returncode
        status_file.write_text(json.dumps(status, indent=2) + '\n')
        if process.returncode:
            raise RuntimeError(f'{job["name"]} failed with {process.returncode}; see its log')
        print('completed', job['name'], flush=True)
        if job['phase'] == 'discovery':
            subprocess.run([sys.executable, str(here / '02_local_model.py'), 'fit', '--root', str(args.output)], check=True)
    if args.phase == 'all':
        subprocess.run([sys.executable, str(here / '02_local_model.py'), 'validate', '--root', str(args.output)], check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/02_local_suite'))
    parser.add_argument('--repeats', type=int, default=24)
    parser.add_argument('--phase', choices=('all', 'discovery', 'validation', 'control'), default='all')
    main(parser.parse_args())
