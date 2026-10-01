"""循环寄存器读取的旧版完整条件：工作集、单双 TC、展开与累加器对照。"""

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

def jobs() -> list[dict]:
    result = []
    for size in (64, 256, 1024, 4096):
        for streams in (1, 4, 8, 16, 32):
            if size < 8 * streams:
                continue
            for cores in (1, 2):
                result.append(dict(family='base', size_kib=size, streams=streams, accumulators=streams, cores=cores, spaces=['vmem', 'cmem']))
    for streams in (16, 32, 64, 128):
        for cores in (1, 2):
            result.append(dict(family='a8', size_kib=1024, streams=streams, accumulators=8, cores=cores, spaces=['vmem', 'cmem']))
            result.append(dict(family='cmem_a2', size_kib=1024, streams=streams, accumulators=2, cores=cores, spaces=['cmem']))
    return result

def counts(job: dict, phase: str) -> list[int]:
    groups = job['size_kib'] // (4 * job['streams'])
    if phase == 'discovery':
        tail = {'base': [16384, 65536, 262144], 'a8': [4096, 16384, 65536], 'cmem_a2': [1024, 4096, 16384]}[job['family']]
        return sorted({1, 3, 17, groups + 3, *tail})
    tail = {'base': [32768, 131072, 524288], 'a8': [8192, 32768, 131072], 'cmem_a2': [2048, 8192, 32768]}[job['family']]
    return sorted({2, 7, 19, 2 * groups + 5, *tail} - set(counts(job, 'discovery')))

def records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    assert data['protocol'] == 'register_cyclic_lcc_v1' and data['baseline_after']
    result = []
    for row in data['records']:
        assert len(row['records']) == 24
        for sample in row['records']:
            assert sample['source_and_all_accumulators_correct']
            assert len(sample['lcc']) == row['cores'] and all(len(values) == 3 for values in sample['lcc'])
            assert [values[2] - values[0] for values in sample['lcc']] == sample['cycles']
        for core in range(row['cores']):
            result.append(dict(space=row['space'], core=core, iterations=row['iterations'], cycles=[sample['cycles'][core] for sample in row['records']]))
    return result

def errors(group: list[dict], model: dict) -> dict:
    # 仅在调用 DMA 残差工具时适配自变量；原始记录和冻结模型明确使用循环次数，不能标成 KiB。
    adapted = [dict(size_bytes=row['iterations'] * 1024, cycles=row['cycles']) for row in group]
    coefficients = dict(intercept_cycles=model['intercept_cycles'], cycles_per_kib=model['cycles_per_iteration'])
    return local.errors(adapted, coefficients)

def freeze(root: Path, name: str, job: dict) -> None:
    group = records(root / f'{name}_discovery' / 'summary.json')
    models = []
    for space, core in sorted({(row['space'], row['core']) for row in group}):
        selected = [row for row in group if row['space'] == space and row['core'] == core]
        n = np.array([row['iterations'] for row in selected])
        y = np.array([np.median(row['cycles']) for row in selected])
        # 候选整数斜率来自发现组；所有原始残差仍保留，不能将取整后公式自动称为精确。
        slope = int(round(np.polyfit(n, y, 1)[0]))
        intercept = float(np.median(y - n * slope))
        model = dict(space=space, core=core, kind='integer_slope_candidate', intercept_cycles=intercept, cycles_per_iteration=slope)
        model['discovery'] = errors(selected, model)
        models.append(model)
    document = dict(job=job, variable='N = cyclic loop iterations, each consumes U vectors', models=models)
    output = root / f'{name}.model.json'
    if output.exists():
        assert json.loads(output.read_text()) == document
    else:
        assert not (root / f'{name}_validation').exists()
        output.write_text(json.dumps(document, indent=2) + '\n')

def validate(root: Path, name: str) -> None:
    document = json.loads((root / f'{name}.model.json').read_text())
    group = records(root / f'{name}_validation' / 'summary.json')
    for model in document['models']:
        selected = [row for row in group if row['space'] == model['space'] and row['core'] == model['core']]
        model['validation'] = errors(selected, model)
    (root / f'{name}.validated.json').write_text(json.dumps(document, indent=2) + '\n')

def table(root: Path) -> None:
    rows = ['| 条件 | 内存 | TC | 周期公式，N=循环次数 | 验证中位数最大误差 | 验证全样本残差 |', '| --- | --- | ---: | --- | ---: | --- |']
    for path in sorted(root.glob('*.validated.json')):
        document = json.loads(path.read_text())
        for model in document['models']:
            check = model['validation']
            formula = f'{model["cycles_per_iteration"]}N + {model["intercept_cycles"]:g}'
            name = path.name.removesuffix('.validated.json')
            rows.append(f'| {name} | {model["space"]} | {model["core"]} | {formula} | {check["max_median_error_cycles"]:.2f} | [{check["min_residual_cycles"]:.2f}, {check["max_residual_cycles"]:.2f}] |')
    (root / 'formulas.md').write_text('\n'.join(rows) + '\n')

def main(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    matrix = jobs()
    (args.output / 'matrix.json').write_text(json.dumps(matrix, indent=2) + '\n')
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(here.parents[1] / 'tpuasm' / 'src')
    for job in matrix:
        name = f'{job["family"]}_s{job["size_kib"]}_u{job["streams"]}_a{job["accumulators"]}_c{job["cores"]}'
        if args.cases and name not in args.cases.split(','):
            continue
        for phase, seed in (('discovery', 541), ('validation', 542)):
            output = args.output / f'{name}_{phase}'
            if phase == 'validation':
                freeze(args.output, name, job)
            if not (output / 'summary.json').exists():
                command = [sys.executable, str(here / '11_register_ring.py'), '--size', str(job['size_kib'] * 1024), '--streams', str(job['streams'])]
                command += ['--accumulators', str(job['accumulators']), '--cores', str(job['cores']), '--spaces', ','.join(job['spaces'])]
                command += ['--iterations', ','.join(map(str, counts(job, phase))), '--seed', str(seed), '--output', str(output)]
                print('starting', name, phase, flush=True)
                with (args.output / f'{name}_{phase}.log').open('w') as stream:
                    subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            records(output / 'summary.json')
        validate(args.output, name)
        table(args.output)
        print('validated', name, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/11_register_suite'))
    parser.add_argument('--cases', help='逗号分隔的配置名，用于先导或分批采样')
    main(parser.parse_args())
