"""为 Host Magic Queue 的大窗口重新采样并独立验证连续分段周期模型。"""

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

refine = import_module('06_host_refine')
magic = import_module('09_magic_model')

def freeze(group: list[dict], window: int) -> dict:
    x = np.array([row['size_bytes'] / 1024 for row in group])
    y = np.array([np.median(row['cycles']) for row in group])
    candidates = []
    for knot in (64, 128, 256, 384, 512, 768):
        design = np.column_stack((np.ones_like(x), x, np.maximum(x - knot, 0)))
        coefficients, _, _, _ = np.linalg.lstsq(design / y[:, None], np.ones_like(y), rcond=None)
        candidates.append((float(np.sum(((design @ coefficients - y) / y)**2)), knot, coefficients))
    _, knot, (intercept, slope, extra) = min(candidates, key=lambda row: row[0])
    return dict(
        kind='empirical_continuous_hinge',
        fit_objective='relative_squared_median_error',
        window=window,
        intercept_cycles=float(intercept),
        cycles_per_kib=float(slope),
        extra_cycles_per_kib=float(extra),
        knot_kib=knot,
        domain_bytes=[int(x.min() * 1024), int(x.max() * 1024)],
    )

def main(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    args.archive.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(args.libtpu_root)
    rows = ['| W | raw 周期中心公式，K=S/1024 | 验证中位数最大误差 | 最大相对误差 | 验证全样本残差 |', '| ---: | --- | ---: | ---: | --- |']
    for window in args.windows:
        model_path = args.output / f'w{window}.model.json'
        for phase, seed in (('discovery', 531), ('validation', 532)):
            sizes = [4, 16, 64, 128, 256, 512, 1024, 2048, 3072, 4096] if phase == 'discovery' else [8, 32, 96, 192, 384, 768, 1536, 2560, 3584]
            if window == 64:
                sizes = [4, 16, 64, 128, 256, 384, 512, 640, 768, 896, 1024] if phase == 'discovery' else [8, 32, 96, 192, 320, 448, 576, 704, 832, 960]
            group = []
            for kib in sizes:
                name = f'{phase}_w{window}_s{kib * 1024}'
                output = args.output / name
                summary = output / 'summary.json'
                if not summary.exists():
                    command = [sys.executable, str(here / '09_magic_host.py'), '--size', str(kib * 1024), '--window', str(window), '--seed', str(seed), '--output', str(output)]
                    print('starting', name, flush=True)
                    with (args.output / f'{name}.log').open('w') as stream:
                        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
                group.append(magic.records(summary))
                shutil.copyfile(summary, args.archive / f'{name}.json')
            if phase == 'discovery':
                model = freeze(group, window)
                model['discovery'] = refine.errors(group, model)
                if model_path.exists():
                    assert json.loads(model_path.read_text()) == model
                else:
                    assert not list(args.output.glob(f'validation_w{window}_*'))
                    model_path.write_text(json.dumps(model, indent=2) + '\n')
                shutil.copyfile(model_path, args.archive / model_path.name)
                print('frozen', model, flush=True)
            else:
                model = json.loads(model_path.read_text())
                model['validation'] = refine.errors(group, model)
                validated = args.output / f'w{window}.validated.json'
                validated.write_text(json.dumps(model, indent=2) + '\n')
                shutil.copyfile(validated, args.archive / validated.name)
                print('validated', model, flush=True)
        sign = '+' if model['cycles_per_kib'] >= 0 else '−'
        formula = f'{model["intercept_cycles"]:.3f} {sign} {abs(model["cycles_per_kib"]):.5f}K + {model["extra_cycles_per_kib"]:.5f}·max(K−{model["knot_kib"]},0)'
        check = model['validation']
        rows.append(f'| {window} | {formula} | {check["max_median_error_cycles"]:.2f} | {check["max_relative_median_error"]:.2%} | [{check["min_residual_cycles"]:.2f}, {check["max_residual_cycles"]:.2f}] |')
        (args.archive / 'formulas.md').write_text('\n'.join(rows) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/09_magic_refine'))
    parser.add_argument('--archive', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/09_magic_refine'))
    parser.add_argument('--libtpu-root', type=Path, default=Path('/tmp/libtpu-0.0.46'))
    parser.add_argument('--windows', type=lambda value: [int(item) for item in value.split(',')], default=[4, 16, 64])
    main(parser.parse_args())
