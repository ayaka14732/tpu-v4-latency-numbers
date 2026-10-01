"""扫描完整 payload RTT：六个跨芯片对、三种独立双 pair、四个片内 TC 对。"""

from __future__ import annotations

import argparse
from importlib import import_module
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

remote = import_module('04_remote_suite')
PROTOCOL = 'remote_lcc_v2_payload_rtt'

def main(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(here.parents[1] / 'tpuasm' / 'src')
    matrix = [dict(name=f'pair_{a}{b}', cores=1, flows=f'{a}:{b}') for a, b in itertools.combinations(range(4), 2)]
    matrix += [dict(name=f'same_chip_{chip}', cores=2, flows=f'{2 * chip}:{2 * chip + 1}') for chip in range(4)]
    for flows in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
        matrix.append(dict(name='parallel_' + '_'.join(f'{a}{b}' for a, b in flows), cores=1, flows=','.join(f'{a}:{b}' for a, b in flows)))
    if args.cases:
        matrix = [job for job in matrix if job['name'] in args.cases.split(',')]
    (args.output / 'matrix.json').write_text(json.dumps(matrix, indent=2) + '\n')
    for job in matrix:
        name = job['name']
        for phase, seed in (('discovery', 491), ('validation', 492)):
            output = args.output / f'{name}_{phase}'
            if phase == 'validation':
                remote.freeze(args.output / f'{name}_discovery', output, args.output / f'{name}.model.json', PROTOCOL)
            sizes = [4096, 16384, 65536, 262144, 1048576, 4194304] if phase == 'discovery' else [8192, 32768, 131072, 524288, 2097152]
            if not (output / 'summary.json').exists():
                command = [sys.executable, str(here / '04_remote_dma.py'), '--round-trip', '--chips', '4', '--size', '4194304', '--sizes', ','.join(map(str, sizes)), '--output', str(output)]
                command += ['--cores', str(job['cores']), '--flows', job['flows'], '--seed', str(seed)]
                print('starting', name, phase, flush=True)
                with (args.output / f'{name}_{phase}.log').open('w') as stream:
                    subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        remote.validate(args.output, name, PROTOCOL)
        remote.archive(args.output, args.archive)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/08_rtt_suite'))
    parser.add_argument('--archive', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/08_rtt'))
    parser.add_argument('--cases', default='')
    main(parser.parse_args())
