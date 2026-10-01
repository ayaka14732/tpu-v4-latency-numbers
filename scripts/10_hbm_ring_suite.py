"""顺序扫描四条 HBM 环形工作集路径、单／双 TC、窗口和独立验证尺寸。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

local = import_module('02_local_model')

def records(path: Path) -> list[dict]:
    document = json.loads(path.read_text())
    assert document['protocol'] == 'local_hbm_ring_lcc_v1' and document['baseline_after']
    assert document['fresh_poison_donated_each_call'] and document['source_unchanged']
    result = []
    for row in document['records']:
        assert row['working_set_bytes_per_core'] == 64 * 1024**2
        assert len(row['records']) == 24
        for sample in row['records']:
            assert sample['full_payload_and_guards_correct']
            assert len(sample['lcc']) == row['cores'] and all(len(values) == 3 for values in sample['lcc'])
            assert [values[2] - values[0] for values in sample['lcc']] == sample['cycles']
        for core in range(row['cores']):
            result.append(dict(path=row['path'], size_bytes=row['size_bytes'], core=core, cycles=[sample['cycles'][core] for sample in row['records']]))
    return result

def freeze(root: Path, name: str) -> None:
    group = records(root / f'{name}_discovery' / 'summary.json')
    models = []
    for path, core in sorted({(row['path'], row['core']) for row in group}):
        selected = [row for row in group if row['path'] == path and row['core'] == core]
        x = [row['size_bytes'] / 1024 for row in selected]
        y = [np.median(row['cycles']) for row in selected]
        slope, intercept = np.polyfit(x, y, 1)
        model = dict(path=path, core=core, kind='empirical_affine', intercept_cycles=float(intercept), cycles_per_kib=float(slope), domain_bytes=[min(x) * 1024, max(x) * 1024])
        model['discovery'] = local.errors(selected, model)
        models.append(model)
    output = root / f'{name}.model.json'
    if output.exists():
        assert json.loads(output.read_text()) == models
    else:
        assert not (root / f'{name}_validation').exists()
        output.write_text(json.dumps(models, indent=2) + '\n')

def validate(root: Path, name: str) -> None:
    models = json.loads((root / f'{name}.model.json').read_text())
    group = records(root / f'{name}_validation' / 'summary.json')
    for model in models:
        selected = [row for row in group if row['path'] == model['path'] and row['core'] == model['core']]
        model['validation'] = local.errors(selected, model)
    (root / f'{name}.validated.json').write_text(json.dumps(models, indent=2) + '\n')

def table(root: Path) -> None:
    rows = ['| TC 数／W | 路径 | TC | 周期中心公式，K=S/1024 | 验证中位数最大误差 | 验证全样本残差 |', '| --- | --- | ---: | --- | ---: | --- |']
    for path in sorted(root.glob('*.validated.json')):
        for model in json.loads(path.read_text()):
            check = model['validation']
            formula = f'{model["intercept_cycles"]:.3f} + {model["cycles_per_kib"]:.5f}K'
            rows.append(f'| {path.stem.split(".")[0]} | {model["path"]} | {model["core"]} | {formula} | {check["max_median_error_cycles"]:.2f} | [{check["min_residual_cycles"]:.2f}, {check["max_residual_cycles"]:.2f}] |')
    (root / 'formulas.md').write_text('\n'.join(rows) + '\n')

def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(here.parents[1] / 'tpuasm' / 'src')
    for cores in args.cores:
        for window in args.windows:
            name = f'c{cores}_w{window}'
            for phase, seed in (('discovery', 511), ('validation', 512)):
                sizes = [4, 16, 64, 256, 1024, 4096] if phase == 'discovery' else [8, 32, 128, 512, 2048]
                if window > 1:
                    sizes = [value for value in sizes if value <= 256]
                output = args.output / f'{name}_{phase}'
                if phase == 'validation':
                    freeze(args.output, name)
                if not (output / 'summary.json').exists():
                    command = [sys.executable, str(here / '10_hbm_ring.py'), '--cores', str(cores), '--window', str(window), '--sizes', ','.join(str(value * 1024) for value in sizes)]
                    command += ['--seed', str(seed), '--output', str(output)]
                    print('starting', name, phase, flush=True)
                    with (args.output / f'{name}_{phase}.log').open('w') as stream:
                        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
                records(output / 'summary.json')
            validate(args.output, name)
            table(args.output)
            print('validated', name, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/10_hbm_ring_suite'))
    parser.add_argument('--cores', type=lambda value: [int(item) for item in value.split(',')], default=[1, 2])
    parser.add_argument('--windows', type=lambda value: [int(item) for item in value.split(',')], default=[1, 4, 8])
    main(parser.parse_args())
