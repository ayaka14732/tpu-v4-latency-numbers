"""依次扫描远程拓扑、窗口与独立验证尺寸；同一时刻只运行一个 TPU 实验。"""

from __future__ import annotations

import argparse
from importlib import import_module
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

local_model = import_module('02_local_model')
PROTOCOL = 'remote_lcc_v3_all_credits_first'

def cases() -> list[dict]:
    result = [dict(name=f'c{a}_c{b}', cores=1, flows=f'{a}:{b}') for a, b in itertools.permutations(range(4), 2)]
    result += [dict(name=f'same_chip_{chip}', cores=2, flows=f'{2 * chip}:{2 * chip + 1}') for chip in range(4)]
    result += [dict(name=f'same_chip_duplex_{chip}', cores=2, flows=f'{2 * chip}:{2 * chip + 1},{2 * chip + 1}:{2 * chip}') for chip in range(4)]
    result += [dict(name='independent', cores=1, flows='0:1,2:3'), dict(name='full_duplex', cores=1, flows='0:1,1:0'), dict(name='same_pair_two_cores', cores=2, flows='0:2,1:3')]
    return result

def records(path: Path, protocol: str = PROTOCOL) -> list[dict]:
    document = json.loads(path.read_text())
    # 单流的两种 Python 循环产生同一操作序列；旧版多流 credit 交错顺序必须重新测量。
    legacy_single = protocol == PROTOCOL and document['protocol'] == 'remote_lcc_v1_poisoned_fixed_stride' and all(len(row['flows']) == 1 for row in document['records'])
    assert (document['protocol'] == protocol or legacy_single) and document['baseline_after']
    for record in document['records']:
        assert record['correct']
        assert record['audit'].get('gap_bundles') is None
        assert record['audit']['boundary'] == 'complete'
        counts = np.asarray(record['lcc'], dtype=np.uint64)
        assert counts.shape == (document['repeats'], len(record['devices']) * record['cores'], 2)
        np.testing.assert_array_equal(counts[:, :, 1] - counts[:, :, 0], record['cycles'])
    return document['records']

def source_records(group: list[dict], source: int) -> list[dict]:
    return [{**record, 'cycles': [sample[source] for sample in record['cycles']]} for record in group]

def freeze(discovery: Path, validation: Path, output: Path, protocol: str = PROTOCOL) -> None:
    group = records(discovery / 'summary.json', protocol)
    models = {}
    for source, _ in group[0]['flows']:
        selected = source_records(group, source)
        x = [record['size_bytes'] / 1024 for record in selected]
        y = [np.median(record['cycles']) for record in selected]
        slope, intercept = np.polyfit(x, y, 1)
        model = {'kind': 'empirical_affine', 'source_node': source, 'intercept_cycles': float(intercept), 'cycles_per_kib': float(slope)}
        model['discovery'] = local_model.errors(selected, model)
        models[str(source)] = model
    document = {'domain_bytes': [min(x) * 1024, max(x) * 1024], 'window': group[0]['window'], 'cores': group[0]['cores'], 'flows': group[0]['flows'], 'models': models}
    if output.exists():
        assert json.loads(output.read_text()) == document, '不能覆盖已冻结公式'
    else:
        assert not validation.exists(), '验证开始后不能反向冻结发现公式'
        output.write_text(json.dumps(document, indent=2) + '\n')

def validate(root: Path, name: str, protocol: str = PROTOCOL) -> None:
    document = json.loads((root / f'{name}.model.json').read_text())
    group = records(root / f'{name}_validation' / 'summary.json', protocol)
    for source, model in document['models'].items():
        model['validation'] = local_model.errors(source_records(group, int(source)), model)
    (root / f'{name}.validated.json').write_text(json.dumps(document, indent=2) + '\n')

def archive(root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    rows = ['| 拓扑 | W | 发起节点 | 计数保存 | 周期中心公式，K=S/1024 | 验证中位数最大误差 | 验证全样本残差 |', '| --- | ---: | ---: | --- | --- | ---: | --- |']
    count = 0
    for path in sorted(root.glob('*.validated.json')):
        name = path.name.removesuffix('.validated.json')
        data = json.loads(path.read_text())
        storage = set()
        for phase in ('discovery', 'validation'):
            summary = root / f'{name}_{phase}' / 'summary.json'
            storage.update(row['audit'].get('counter_storage', 'registers') for row in json.loads(summary.read_text())['records'])
            shutil.copyfile(summary, destination / f'{name}_{phase}.json')
        assert len(storage) == 1
        counter_storage, = storage
        for ending in ('model.json', 'validated.json'):
            shutil.copyfile(root / f'{name}.{ending}', destination / f'{name}.{ending}')
        for model in data['models'].values():
            check = model['validation']
            rows.append(f'| {name} | {data["window"]} | {model["source_node"]} | {counter_storage} | {model["intercept_cycles"]:.3f} + {model["cycles_per_kib"]:.5f}K | {check["max_median_error_cycles"]:.2f} | [{check["min_residual_cycles"]:.2f}, {check["max_residual_cycles"]:.2f}] |')
            count += 1
    (destination / 'formulas.md').write_text('\n'.join(rows) + '\n')
    print(f'archived {count} source formulas', flush=True)

def main(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(here.parents[1] / 'tpuasm' / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    selected = [case for case in cases() if (not args.cases or case['name'] in args.cases.split(',')) and (args.space == 'vmem' or case['cores'] == 1)]
    matrix = [{**case, 'window': window} for window in args.windows for case in selected]
    (root / 'matrix.json').write_text(json.dumps(matrix, indent=2) + '\n')
    for job in matrix:
        name = f'{job["name"]}_w{job["window"]}'
        capacity = 4194304 if job['window'] == 1 else 262144
        for phase, seed in (('discovery', 431), ('validation', 432)):
            output = root / f'{name}_{phase}'
            if phase == 'validation':
                freeze(root / f'{name}_discovery', output, root / f'{name}.model.json')
            sizes = ([4096, 16384, 65536, 262144, 1048576, 4194304] if job['window'] == 1 else [4096, 16384, 65536, 131072, 262144]) if phase == 'discovery' else (
                [8192, 32768, 131072, 524288, 2097152] if job['window'] == 1 else [8192, 32768, 98304, 196608])
            status_path = root / f'{name}_{phase}.status.json'
            if status_path.exists() and json.loads(status_path.read_text())['exit_code'] == 0:
                continue
            command = [sys.executable, str(here / '04_remote_dma.py'), '--output', str(output), '--size', str(capacity), '--sizes', ','.join(map(str, sizes)), '--chips', '4', '--space', args.space]
            for key, value in (('cores', job['cores']), ('flows', job['flows']), ('window', job['window']), ('seed', seed), ('repeats', args.repeats)):
                command += ['--' + key, str(value)]
            print('starting', name, phase, flush=True)
            status = {'command': command, 'exit_code': None}
            status_path.write_text(json.dumps(status, indent=2) + '\n')
            with (root / f'{name}_{phase}.log').open('w') as stream:
                status['exit_code'] = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT).returncode
            status_path.write_text(json.dumps(status, indent=2) + '\n')
            if status['exit_code']:
                raise RuntimeError(f'{name} {phase} failed; see its log')
        validate(root, name)
        archive(root, args.archive)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/04_remote_suite'))
    parser.add_argument('--archive', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/04_remote'))
    parser.add_argument('--cases', default='')
    parser.add_argument('--space', choices=('vmem', 'cmem'), default='vmem')
    parser.add_argument('--windows', type=lambda value: [int(item) for item in value.split(',')], default=[1, 4, 8])
    parser.add_argument('--repeats', type=int, default=24)
    main(parser.parse_args())
