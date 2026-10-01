"""顺序测量 Host Magic Queue 的 BC-LCC 包围区间，覆盖 W=1/4/16/64 与独立尺寸。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

local = import_module('02_local_model')

def main(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    args.archive.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(args.libtpu_root)
    for window in args.windows:
        for phase, seed in (('control', 500), ('discovery', 501), ('validation', 502)):
            if phase == 'control':
                sizes = [4]
            elif window == 64:
                sizes = [4, 16, 64, 256, 1024] if phase == 'discovery' else [8, 32, 128, 512]
            else:
                sizes = [4, 16, 64, 256, 1024, 4096] if phase == 'discovery' else [8, 32, 128, 512, 2048]
            group = []
            for kib in sizes:
                name = f'{phase}_w{window}_s{kib * 1024}'
                output = args.output / name
                if not (output / 'summary.json').exists():
                    command = [sys.executable, str(here / '09_magic_host.py'), '--size', str(kib * 1024), '--window', str(window), '--seed', str(seed), '--output', str(output)]
                    if phase == 'control':
                        command += ['--empty']
                    print('starting', name, flush=True)
                    with (args.output / f'{name}.log').open('w') as stream:
                        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
                data = json.loads((output / 'summary.json').read_text())
                assert data['protocol'] == 'host-magic-bc-lcc-envelope-v2' and data['counter_words_read_after_end'] and data['cleanup_clean']
                assert data['empty_window'] == (phase == 'control') and len(data['records']) == 24
                for row in data['records']:
                    assert row['full_payload_and_guards_correct'] and row['lcc'][1] - row['lcc'][0] == row['cycles']
                group.append({'size_bytes': data['size_bytes'], 'cycles': [row['cycles'] for row in data['records']]})
                shutil.copyfile(output / 'summary.json', args.archive / f'{name}.json')
            if phase == 'discovery':
                x = [row['size_bytes'] / 1024 for row in group]
                y = [np.median(row['cycles']) for row in group]
                slope, intercept = np.polyfit(x, y, 1)
                model = dict(window=window, kind='empirical_affine', intercept_cycles=float(intercept), cycles_per_kib=float(slope), domain_bytes=[int(min(x) * 1024), int(max(x) * 1024)])
                model['discovery'] = local.errors(group, model)
                path = args.output / f'w{window}.model.json'
                if path.exists():
                    assert json.loads(path.read_text()) == model
                else:
                    assert not list(args.output.glob(f'validation_w{window}_*'))
                    path.write_text(json.dumps(model, indent=2) + '\n')
                shutil.copyfile(path, args.archive / path.name)
            elif phase == 'validation':
                model['validation'] = local.errors(group, model)
                path = args.output / f'w{window}.validated.json'
                path.write_text(json.dumps(model, indent=2) + '\n')
                shutil.copyfile(path, args.archive / path.name)
                print('validated', model, flush=True)
    shutil.copyfile(args.output / 'control_w1_s4096' / 'counter.bca', args.archive / 'counter.bca')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/09_magic_suite'))
    parser.add_argument('--archive', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/09_magic_host'))
    parser.add_argument('--libtpu-root', type=Path, default=Path('/tmp/libtpu-0.0.46'))
    parser.add_argument('--windows', type=lambda value: [int(item) for item in value.split(',')], default=[1, 4, 16, 64])
    main(parser.parse_args())
