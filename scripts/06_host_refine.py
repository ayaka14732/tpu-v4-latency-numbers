"""为 Host W=4/16 的大 payload 区间增加独立发现和验证，冻结连续分段周期模型。"""

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

def freeze(records: list[dict], window: int) -> dict:
    x = np.array([record['size_bytes'] / 1024 for record in records])
    y = np.array([np.median(record['cycles']) for record in records])
    candidates = []
    for knot in (1024, 1536, 2048, 2560, 3072):
        design = np.column_stack((np.ones_like(x), x, np.maximum(x - knot, 0)))
        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        error = np.sum((design @ coefficients - y) ** 2)
        candidates.append((float(error), knot, coefficients))
    _, knot, (intercept, slope, extra) = min(candidates, key=lambda row: row[0])
    return {
        'kind': 'empirical_continuous_hinge',
        'window': window,
        'domain_bytes': [int(x.min() * 1024), int(x.max() * 1024)],
        'intercept_cycles': float(intercept),
        'cycles_per_kib': float(slope),
        'knot_kib': knot,
        'extra_cycles_per_kib': float(extra),
    }

def errors(records: list[dict], model: dict) -> dict:
    # 转换只用于复用统一残差统计；不改写任何归档原始读数。
    adjusted = []
    relative = []
    for record in records:
        x = record['size_bytes'] / 1024
        correction = model['extra_cycles_per_kib'] * max(x - model['knot_kib'], 0)
        adjusted.append({**record, 'cycles': (np.asarray(record['cycles']) - correction).tolist()})
        prediction = model['intercept_cycles'] + model['cycles_per_kib'] * x + correction
        median = float(np.median(record['cycles']))
        relative.append(abs(prediction - median) / median)
    result = local.errors(adjusted, model)
    result['max_relative_median_error'] = max(relative)
    return result

def main(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    args.archive.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(here.parents[1] / 'tpuasm' / 'src')
    for window in args.windows:
        model_path = args.output / f'w{window}.model.json'
        for phase, seed in (('discovery', 471), ('validation', 472)):
            sizes = [4, 64, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096] if phase == 'discovery' else [8, 128, 768, 1280, 1792, 2304, 2816, 3328, 3840]
            group = []
            for kib in sizes:
                name = f'{phase}_w{window}_s{kib * 1024}'
                output = args.output / name
                summary = output / 'summary.json'
                if not summary.exists():
                    command = [sys.executable, str(here / '06_host_dma.py'), '--size', str(kib * 1024), '--window', str(window), '--seed', str(seed), '--output', str(output)]
                    print('starting', name, flush=True)
                    with (args.output / f'{name}.log').open('w') as stream:
                        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
                data = json.loads(summary.read_text())
                assert data['baseline_after'] and data['poison_input_unchanged'] and len(data['records']) == 24
                for record in data['records']:
                    assert record['full_payload_correct'] and record['pinned_host_verified'] and record['lcc'][1] - record['lcc'][0] == record['cycles']
                group.append({'size_bytes': data['size_bytes'], 'cycles': [record['cycles'] for record in data['records']]})
                shutil.copyfile(summary, args.archive / f'{name}.json')
            if phase == 'discovery':
                model = freeze(group, window)
                model['discovery'] = errors(group, model)
                if model_path.exists():
                    assert json.loads(model_path.read_text()) == model
                else:
                    assert not list(args.output.glob(f'validation_w{window}_*'))
                    model_path.write_text(json.dumps(model, indent=2) + '\n')
                shutil.copyfile(model_path, args.archive / model_path.name)
                print('frozen', model, flush=True)
            else:
                model = json.loads(model_path.read_text())
                model['validation'] = errors(group, model)
                validated = args.output / f'w{window}.validated.json'
                validated.write_text(json.dumps(model, indent=2) + '\n')
                shutil.copyfile(validated, args.archive / validated.name)
                print('validated', model, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/06_host_refine'))
    parser.add_argument('--archive', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/06_host_refine'))
    parser.add_argument('--windows', type=lambda value: [int(item) for item in value.split(',')], default=[4, 16])
    main(parser.parse_args())
