"""顺序运行 TC VMEM／Megacore Shared CMEM→BC 私有 BMEM 的周期矩阵。"""

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
    for space in args.spaces:
        for window in args.windows:
            model_path = args.output / f'{space}_w{window}.model.json'
            for phase, seed in (('discovery', 481), ('validation', 482)):
                if window == 1:
                    sizes = [4, 16, 64, 128, 512, 2048] if phase == 'discovery' else [8, 32, 96, 256, 1024]
                elif window == 4:
                    sizes = [4, 16, 64, 128, 256, 512] if phase == 'discovery' else [8, 32, 96, 192, 384]
                else:
                    sizes = [4, 16, 64, 128] if phase == 'discovery' else [8, 32, 96]
                group = []
                for kib in sizes:
                    name = f'{space}_{phase}_w{window}_s{kib * 1024}'
                    output = args.output / name
                    summary = output / 'summary.json'
                    if not summary.exists():
                        command = [sys.executable, str(here / '07_bc_pull.py'), '--size', str(kib * 1024), '--window', str(window), '--space', space, '--seed', str(seed), '--output', str(output)]
                        print('starting', name, flush=True)
                        with (args.output / f'{name}.log').open('w') as stream:
                            subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
                    data = json.loads(summary.read_text())
                    assert data['residents_stopped'] and data['hbm_freed'] and data['observer_removed'] and len(data['records']) == 24
                    for row in data['records']:
                        assert row['full_payload_and_guards_correct'] and row['tc_source_released'] and row['lcc'][1] - row['lcc'][0] == row['cycles']
                    group.append({'size_bytes': data['size_bytes'], 'cycles': [row['cycles'] for row in data['records']]})
                    shutil.copyfile(summary, args.archive / f'{name}.json')
                if phase == 'discovery':
                    x = [row['size_bytes'] / 1024 for row in group]
                    y = [np.median(row['cycles']) for row in group]
                    slope, intercept = np.polyfit(x, y, 1)
                    model = dict(space=space, window=window, kind='empirical_affine', intercept_cycles=float(intercept), cycles_per_kib=float(slope), domain_bytes=[int(min(x) * 1024), int(max(x) * 1024)])
                    model['discovery'] = local.errors(group, model)
                    if model_path.exists():
                        assert json.loads(model_path.read_text()) == model
                    else:
                        assert not list(args.output.glob(f'{space}_validation_w{window}_*'))
                        model_path.write_text(json.dumps(model, indent=2) + '\n')
                    shutil.copyfile(model_path, args.archive / model_path.name)
                else:
                    model['validation'] = local.errors(group, model)
                    path = args.output / f'{space}_w{window}.validated.json'
                    path.write_text(json.dumps(model, indent=2) + '\n')
                    shutil.copyfile(path, args.archive / path.name)
                    source = args.output / f'{space}_discovery_w{window}_s4096'
                    shutil.copyfile(source / 'program.bca', args.archive / f'{space}_w{window}_4k.bca')
                    if space == 'cmem':
                        for ending in ('tpuasm', 'json'):
                            shutil.copyfile(source / f'cmem_source.{ending}', args.archive / f'cmem_w{window}_source.{ending}')
                    print('validated', model, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/07_bc_suite'))
    parser.add_argument('--archive', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/07_bc_pull'))
    parser.add_argument('--libtpu-root', type=Path, default=Path('/tmp/libtpu-0.0.46'))
    parser.add_argument('--windows', type=lambda value: [int(item) for item in value.split(',')], default=[1, 4, 16])
    parser.add_argument('--spaces', type=lambda value: value.split(','), default=['vmem', 'cmem'])
    main(parser.parse_args())
