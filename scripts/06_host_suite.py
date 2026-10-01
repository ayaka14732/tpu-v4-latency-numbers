"""顺序运行 TC 发起的 HBM→pinned Host DMA 周期矩阵。"""

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
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    args.archive.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(here.parents[1] / 'tpuasm' / 'src')
    for window in args.windows:
        group = {}
        for phase, seed in (('discovery', 461), ('validation', 462)):
            sizes = [4, 16, 64, 256, 1024, 4096] if phase == 'discovery' else [8, 32, 128, 512, 2048]
            group[phase] = []
            for kib in sizes:
                name = f'{phase}_w{window}_s{kib * 1024}'
                output = root / name
                status_path = root / f'{name}.status.json'
                if not status_path.exists() or json.loads(status_path.read_text())['exit_code'] != 0:
                    command = [sys.executable, str(here / '06_host_dma.py'), '--size', str(kib * 1024), '--window', str(window), '--seed', str(seed), '--output', str(output)]
                    print('starting', name, flush=True)
                    status = {'command': command, 'exit_code': None}
                    status_path.write_text(json.dumps(status, indent=2) + '\n')
                    with (root / f'{name}.log').open('w') as stream:
                        status['exit_code'] = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT).returncode
                    status_path.write_text(json.dumps(status, indent=2) + '\n')
                    if status['exit_code']:
                        raise RuntimeError(f'{name} failed; see log')
                data = json.loads((output / 'summary.json').read_text())
                assert data['baseline_after'] and data['poison_input_unchanged'] and len(data['records']) == 24
                for record in data['records']:
                    assert record['full_payload_correct'] and record['pinned_host_verified'] and record['lcc'][1] - record['lcc'][0] == record['cycles']
                group[phase].append({'size_bytes': data['size_bytes'], 'cycles': [record['cycles'] for record in data['records']]})
                shutil.copyfile(output / 'summary.json', args.archive / f'{name}.json')
            if phase == 'discovery':
                x = [record['size_bytes'] / 1024 for record in group[phase]]
                y = [np.median(record['cycles']) for record in group[phase]]
                slope, intercept = np.polyfit(x, y, 1)
                model = {'window': window, 'kind': 'empirical_affine', 'intercept_cycles': float(intercept), 'cycles_per_kib': float(slope), 'domain_bytes': [int(min(x) * 1024), int(max(x) * 1024)]}
                model['discovery'] = local.errors(group[phase], model)
                path = root / f'w{window}.model.json'
                if path.exists():
                    assert json.loads(path.read_text()) == model, '冻结后的公式不可覆盖'
                else:
                    assert not list(root.glob(f'validation_w{window}_*')), '验证开始后不能反向冻结模型'
                    path.write_text(json.dumps(model, indent=2) + '\n')
                shutil.copyfile(path, args.archive / path.name)
        model['validation'] = local.errors(group['validation'], model)
        (root / f'w{window}.validated.json').write_text(json.dumps(model, indent=2) + '\n')
        shutil.copyfile(root / f'w{window}.validated.json', args.archive / f'w{window}.validated.json')
        shutil.copyfile(root / f'discovery_w{window}_s4096' / 'instrumented.tpuasm', args.archive / f'w{window}_4k.tpuasm')
        print('validated', window, model, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/06_host_suite'))
    parser.add_argument('--archive', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/06_host'))
    parser.add_argument('--windows', type=lambda value: [int(item) for item in value.split(',')], default=[1, 4, 16])
    main(parser.parse_args())
