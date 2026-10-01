"""冻结本地 DMA 的周期域模型，再用独立进程和未参与建模的尺寸检验残差。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

def load_phase(root: Path, phase: str, ready_only: bool = False) -> list[dict]:
    jobs = [job for job in json.loads((root / 'matrix.json').read_text()) if job['phase'] == phase]
    records = []
    for job in jobs:
        status_path = root / f'{job["name"]}.status.json'
        if ready_only and (not status_path.exists() or json.loads(status_path.read_text())['exit_code'] is None):
            continue
        status = json.loads(status_path.read_text())
        assert status['exit_code'] == 0, job['name']
        data = json.loads((root / job['name'] / 'summary.json').read_text())
        assert data['environment']['protocol'] == 'local_dma_v2_poisoned_destination'
        if job['cores'] == 2:
            assert data['environment']['pre_measurement_rendezvous']
        assert len(data['records']) == 6 * len(job['sizes']), job['name']
        observed = set()
        for record in data['records']:
            assert record['correct'] and record['cores'] == job['cores'] and record['window'] == job['window']
            assert record['boundary'] == job['boundary']
            key = record['path'], record['size_bytes']
            assert key not in observed and record['size_bytes'] in job['sizes'], (job, key)
            observed.add(key)
            counts = np.asarray(record['lcc'], dtype=np.uint64)
            gaps = np.asarray(record['gaps_from_begin'], dtype=np.uint64)
            np.testing.assert_array_equal(counts[:, :, 1:] - counts[:, :, :1], gaps)
            assert counts.shape == (data['environment']['repeats'], job['cores'], 3)
            for core in range(job['cores']):
                records.append({**record, 'core': core, 'cycles': gaps[:, core, 1].tolist(), 'issue_cycles': gaps[:, core, 0].tolist(), 'job': job['name']})
        paths = ('hbm_vmem', 'vmem_hbm', 'hbm_cmem', 'cmem_hbm', 'cmem_vmem', 'vmem_cmem')
        assert observed == {(path, size) for path in paths for size in job['sizes']}
    return records

def model_key(record: dict) -> str:
    return f'{record["path"]}_c{record["cores"]}_tc{record["core"]}_w{record["window"]}'

def errors(records: list[dict], model: dict) -> dict:
    residuals, median_residuals = [], []
    for record in records:
        prediction = model['intercept_cycles'] + model['cycles_per_kib'] * record['size_bytes'] / 1024
        values = np.array(record['cycles'], dtype=float) - prediction
        residuals.extend(values.tolist())
        median_residuals.append(float(np.median(values)))
    values = np.asarray(residuals)
    return {
        'configurations': len(records),
        'samples': len(values),
        'min_residual_cycles': float(values.min()),
        'p05_residual_cycles': float(np.quantile(values, 0.05)),
        'p50_residual_cycles': float(np.quantile(values, 0.5)),
        'p95_residual_cycles': float(np.quantile(values, 0.95)),
        'max_residual_cycles': float(values.max()),
        'median_mae_cycles': float(np.mean(np.abs(median_residuals))),
        'max_median_error_cycles': float(np.max(np.abs(median_residuals))),
    }

def fit(root: Path) -> None:
    records = load_phase(root, 'discovery', ready_only=True)
    previous = json.loads((root / 'models.json').read_text())['models'] if (root / 'models.json').exists() else {}
    models = {}
    for key in sorted({model_key(record) for record in records}):
        group = [record for record in records if model_key(record) == key and record['size_bytes'] >= 4096]
        x = np.array([record['size_bytes'] / 1024 for record in group])
        y = np.array([np.median(record['cycles']) for record in group])
        assert len(set(x)) >= 4, key
        # 这里的拟合只描述已经直接量到的 raw 完成周期随字节数的中心趋势；
        # 不拟合 host 时间与循环次数，也不把点模型当作逐周期确定性的硬件规律。
        slope, intercept = np.polyfit(x, y, 1)
        kind = 'empirical_affine'
        if group[0]['cores'] == 1 and group[0]['path'] in ('cmem_vmem', 'vmem_cmem'):
            slope = group[0]['window'] * (0.5 if group[0]['path'] == 'cmem_vmem' else 1)
            intercept = float(np.median(y - slope * x))
            kind = 'fixed_payload_rate'
        model = {
            'kind': kind,
            'path': group[0]['path'],
            'cores': group[0]['cores'],
            'core': group[0]['core'],
            'window': group[0]['window'],
            'intercept_cycles': round(float(intercept), 6),
            'cycles_per_kib': round(float(slope), 6),
            'size_domain_bytes': [int(min(x) * 1024), int(max(x) * 1024)],
        }
        model['formula'] = f'C(S) = {model["intercept_cycles"]:g} + {model["cycles_per_kib"]:g} * (S / 1024)'
        model['discovery'] = errors(group, model)
        if key in previous:
            assert model == previous[key], f'发现数据变化，不可覆盖已冻结模型：{key}'
        else:
            validation_name = f'validation_c{model["cores"]}_w{model["window"]}'
            assert not (root / validation_name / 'results.jsonl').exists(), f'验证已开始，不能事后冻结 {key}'
        models[key] = model
    output = {'units': 'local LCC cycles; S is bytes per DMA; C covers the entire W-message burst', 'models': models}
    (root / 'models.json').write_text(json.dumps(output, indent=2) + '\n')
    print(f'froze {len(models)} cycle models', flush=True)

def validate(root: Path) -> None:
    document = json.loads((root / 'models.json').read_text())
    records = load_phase(root, 'validation')
    assert set(document['models']) == {model_key(record) for record in records}
    controls = load_phase(root, 'control')
    for record in controls:
        assert set(record['issue_cycles']) == {2} and set(record['cycles']) == {4}, record
    for key, model in document['models'].items():
        group = [record for record in records if model_key(record) == key]
        assert group and all(model['size_domain_bytes'][0] <= record['size_bytes'] <= model['size_domain_bytes'][1] for record in group)
        model['validation'] = errors(group, model)
    document['issue_only_control'] = {'configurations': len(controls), 'all_R1': 2, 'all_R2': 4}
    (root / 'validated_models.json').write_text(json.dumps(document, indent=2) + '\n')
    lines = ['| 路径 | TC 数 / 本地 TC | W | raw 周期中心公式，S 为每条字节数 | 验证组中位数最大误差 | 验证组全样本残差范围 |', '| --- | --- | ---: | --- | ---: | --- |']
    for model in document['models'].values():
        check = model['validation']
        lines.append(f'| `{model["path"]}` | {model["cores"]} / {model["core"]} | {model["window"]} | `{model["formula"]}` | {check["max_median_error_cycles"]:.2f} | [{check["min_residual_cycles"]:.2f}, {check["max_residual_cycles"]:.2f}] |')
    (root / 'formulas.md').write_text('\n'.join(lines) + '\n')
    print(f'validated {len(document["models"])} models; {len(controls)} issue-only controls', flush=True)

def archive(root: Path, destination: Path) -> None:
    validate(root)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('matrix.json', 'models.json', 'validated_models.json', 'formulas.md'):
        shutil.copyfile(root / name, destination / name)
    for job in json.loads((root / 'matrix.json').read_text()):
        output = destination / job['name']
        output.mkdir(exist_ok=True)
        for name in ('environment.json', 'summary.json'):
            shutil.copyfile(root / job['name'] / name, output / name)
        shutil.copyfile(root / f'{job["name"]}.status.json', destination / f'{job["name"]}.status.json')
        # 留一个完整 CMEM → TC VMEM 程序供审阅，其余配置均由脚本和 matrix 重建。
        name = f'cmem_vmem-{job["sizes"][0]}-w{job["window"]}-c{job["cores"]}-{job["boundary"]}'
        for suffix in ('.tpuasm', '.full.tpuasm'):
            shutil.copyfile(root / job['name'] / (name + suffix), output / (name + suffix))
    print(f'archived raw counters, models, status and representative listings to {destination}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fit', 'validate', 'archive'))
    parser.add_argument('--root', type=Path, default=Path('/tmp/tpu_latency_numbers/02_local_suite'))
    parser.add_argument('--destination', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/02_local'))
    args = parser.parse_args()
    if args.action == 'archive':
        archive(args.root, args.destination)
    elif args.action == 'fit':
        fit(args.root)
    else:
        validate(args.root)
