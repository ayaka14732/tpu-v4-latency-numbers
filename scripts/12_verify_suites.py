"""仅用 CPU 重放远程、环形 HBM 与循环寄存器矩阵，核对覆盖范围和冻结模型。"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from importlib import import_module
import json
from pathlib import Path

import numpy as np

remote = import_module('04_remote_suite')
ring = import_module('10_hbm_ring_suite')
registers = import_module('11_register_suite')

def check_errors(records: list[dict], model: dict, phase: str, measure: Callable[[list[dict], dict], dict] = remote.local_model.errors) -> int:
    errors = measure(records, model)
    assert set(errors) == set(model[phase])
    for key, value in errors.items():
        np.testing.assert_allclose(value, model[phase][key], rtol=1e-12, atol=1e-8)
    return errors['samples']

def verify_length_edits(row: dict) -> None:
    count = len(row['flows']) * row['window'] * (2 if row.get('round_trip', False) else 1)
    assert row['audit']['remote_dma_count'] == count
    edits = row['length_edits']
    if row['size_bytes'] == row['capacity_bytes']:
        assert not edits
        return
    assert Counter(edit['mnemonic'] for edit in edits) == {'dma.general': count, 'vwait.ge': 2 * count, 'vsyncadd.s32': 2 * count}
    capacity, size = row['capacity_bytes'] // 512, row['size_bytes'] // 512
    for edit in edits:
        expected = list(edit['before'])
        if edit['mnemonic'] == 'dma.general':
            assert expected[2] == f'length={capacity}'
            expected[2] = f'length={size}'
        else:
            sign = 1 if edit['mnemonic'] == 'vwait.ge' else -1
            assert expected[1] == str(sign * capacity)
            expected[1] = str(sign * size)
        assert edit['after'] == expected, '大小扫描不得改变地址、flag 或其他操作数'

def verify_remote(root: Path, space: str, partial: bool) -> None:
    expected = {f'{case["name"]}_w{window}': case for case in remote.cases() for window in (1, 4, 8) if space == 'vmem' or case['cores'] == 1}
    actual = {path.name.removesuffix('.validated.json') for path in root.glob('*.validated.json')}
    assert actual <= expected.keys()
    assert partial or actual == expected.keys(), f'{space}: 缺少 {sorted(expected.keys() - actual)}'
    intervals = 0
    for name in sorted(actual):
        document = json.loads((root / f'{name}.validated.json').read_text())
        frozen = json.loads((root / f'{name}.model.json').read_text())
        window = int(name.rsplit('_w', 1)[1])
        assert document['window'] == window
        case = expected[name]
        assert document['cores'] == case['cores']
        assert document['flows'] == [list(map(int, flow.split(':'))) for flow in case['flows'].split(',')]
        assert {int(source) for source in document['models']} == {source for source, _ in document['flows']}
        for key, model in document['models'].items():
            assert {name: value for name, value in model.items() if name != 'validation'} == frozen['models'][key]
        for phase in ('discovery', 'validation'):
            group = remote.records(root / f'{name}_{phase}.json')
            # 初始 TC VMEM 单窗口记录早于 CMEM 路径，没有 space 字段；当时只有 vmem。
            assert all(row.get('space', 'vmem') == space for row in group)
            sizes = ([4, 16, 64, 256, 1024, 4096] if window == 1 else [4, 16, 64, 128, 256]) if phase == 'discovery' else (
                [8, 32, 128, 512, 2048] if window == 1 else [8, 32, 96, 192])
            assert {row['size_bytes'] for row in group} == {size * 1024 for size in sizes}
            assert all(row['window'] == window and row['flows'] == document['flows'] and row['cores'] == document['cores'] for row in group)
            assert all(row['capacity_bytes'] == (4194304 if window == 1 else 262144) and len(row['devices']) == 4 for row in group)
            assert all(len(row['cycles']) == 24 for row in group)
            for row in group:
                verify_length_edits(row)
                audit = row['audit']
                storage = audit.get('counter_storage', 'registers')
                assert storage in ('registers', 'smem')
                assert len(audit['counter_sregs']) == (4 if storage == 'registers' else 2)
            for source, model in document['models'].items():
                intervals += check_errors(remote.source_records(group, int(source)), model, phase)
    print(space, f'{len(actual)}/{len(expected)}', 'remote groups,', intervals, 'source intervals verified')

def verify_ring(root: Path, partial: bool) -> None:
    expected = {f'c{cores}_w{window}' for cores in (1, 2) for window in (1, 4, 8)}
    actual = {path.name.removesuffix('.validated.json') for path in root.glob('*.validated.json')}
    assert actual <= expected
    assert partial or actual == expected, f'HBM ring: 缺少 {sorted(expected - actual)}'
    intervals = 0
    for name in sorted(actual):
        models = json.loads((root / f'{name}.validated.json').read_text())
        frozen = json.loads((root / f'{name}.model.json').read_text())
        assert [{key: value for key, value in model.items() if key != 'validation'} for model in models] == frozen
        cores, window = int(name[1]), int(name.rsplit('_w', 1)[1])
        assert {(model['path'], model['core']) for model in models} == {(path, core) for path in ('hbm_vmem', 'vmem_hbm', 'hbm_cmem', 'cmem_hbm') for core in range(cores)}
        for phase in ('discovery', 'validation'):
            path = root / f'{name}_{phase}' / 'summary.json'
            raw = json.loads(path.read_text())
            sizes = [4, 16, 64, 256, 1024, 4096] if phase == 'discovery' else [8, 32, 128, 512, 2048]
            if window > 1:
                sizes = [size for size in sizes if size <= 256]
            assert {(row['path'], row['size_bytes']) for row in raw['records']} == {(path, size * 1024) for path in ('hbm_vmem', 'vmem_hbm', 'hbm_cmem', 'cmem_hbm') for size in sizes}
            assert all(row['cores'] == cores and row['window'] == window for row in raw['records'])
            group = ring.records(path)
            for model in models:
                selected = [row for row in group if row['path'] == model['path'] and row['core'] == model['core']]
                intervals += check_errors(selected, model, phase)
    print('HBM ring', f'{len(actual)}/{len(expected)}', 'groups,', intervals, 'TC intervals verified')

def verify_rtt(root: Path, partial: bool) -> None:
    expected = {'pair_01', 'pair_02', 'pair_03', 'pair_12', 'pair_13', 'pair_23', 'parallel_01_23', 'parallel_02_13', 'parallel_03_12'}
    expected |= {f'same_chip_{chip}' for chip in range(4)}
    actual = {path.name.removesuffix('.validated.json') for path in root.glob('*.validated.json')}
    assert actual <= expected
    assert partial or actual == expected, f'RTT: 缺少 {sorted(expected - actual)}'
    intervals = 0
    for name in sorted(actual):
        document = json.loads((root / f'{name}.validated.json').read_text())
        frozen = json.loads((root / f'{name}.model.json').read_text())
        if name.startswith('same_chip_'):
            chip = int(name.rsplit('_', 1)[1])
            cores, flows = 2, [[2 * chip, 2 * chip + 1]]
        else:
            cores, flows = 1, [[int(pair[0]), int(pair[1])] for pair in name.split('_')[1:]]
        assert document['cores'] == cores and document['flows'] == flows and document['window'] == 1
        assert {int(source) for source in document['models']} == {source for source, _ in flows}
        for source, model in document['models'].items():
            assert {key: value for key, value in model.items() if key != 'validation'} == frozen['models'][source]
        for phase in ('discovery', 'validation'):
            group = remote.records(root / f'{name}_{phase}.json', 'remote_lcc_v2_payload_rtt')
            assert all(row['round_trip'] and row['space'] == 'vmem' for row in group)
            assert all(row['cores'] == cores and row['flows'] == flows and row['window'] == 1 for row in group)
            assert all(row['capacity_bytes'] == 4194304 and len(row['devices']) == 4 for row in group)
            assert all(len(row['cycles']) == 24 for row in group)
            sizes = [4096, 16384, 65536, 262144, 1048576, 4194304] if phase == 'discovery' else [8192, 32768, 131072, 524288, 2097152]
            assert {row['size_bytes'] for row in group} == set(sizes)
            for row in group:
                verify_length_edits(row)
            for source, model in document['models'].items():
                intervals += check_errors(remote.source_records(group, int(source)), model, phase)
    print('RTT', f'{len(actual)}/{len(expected)}', 'topologies,', intervals, 'source intervals verified')

def verify_registers(root: Path, partial: bool) -> None:
    expected = {f'{job["family"]}_s{job["size_kib"]}_u{job["streams"]}_a{job["accumulators"]}_c{job["cores"]}': job for job in registers.jobs()}
    actual = {path.name.removesuffix('.validated.json') for path in root.glob('*.validated.json')}
    assert actual <= expected.keys()
    assert partial or actual == expected.keys(), f'register ring: 缺少 {sorted(expected.keys() - actual)}'
    intervals = 0
    for name in sorted(actual):
        document = json.loads((root / f'{name}.validated.json').read_text())
        frozen = json.loads((root / f'{name}.model.json').read_text())
        assert document['job'] == expected[name] == frozen['job']
        assert [{key: value for key, value in model.items() if key != 'validation'} for model in document['models']] == frozen['models']
        job = expected[name]
        assert {(model['space'], model['core']) for model in document['models']} == {(space, core) for space in job['spaces'] for core in range(job['cores'])}
        for phase in ('discovery', 'validation'):
            path = root / f'{name}_{phase}' / 'summary.json'
            raw = json.loads(path.read_text())
            assert {(row['space'], row['iterations']) for row in raw['records']} == {(space, count) for space in job['spaces'] for count in registers.counts(job, phase)}
            assert all(row['size_bytes'] == job['size_kib'] * 1024 and row['streams'] == job['streams'] and row['accumulators'] == job['accumulators'] and row['cores'] == job['cores'] for row in raw['records'])
            group = registers.records(path)
            for model in document['models']:
                selected = [row for row in group if row['space'] == model['space'] and row['core'] == model['core']]
                intervals += check_errors(selected, model, phase, registers.errors)
    print('register ring', f'{len(actual)}/{len(expected)}', 'groups,', intervals, 'TC intervals verified')

def verify_legacy_scope(root: Path, partial: bool) -> None:
    legacy = json.loads((Path(__file__).resolve().parent / 'support' / 'legacy_scope.json').read_text())
    comparisons = {}
    for family, directory in (('hbm', '05_bc'), ('vmem', '07_bc_pull'), ('cmem', '07_bc_pull'), ('tc_host', '06_host'), ('magic', '09_magic_host')):
        expected = {tuple(row) for row in legacy['bc_host'][family]}
        actual = set()
        for path in (root / 'evidence' / directory).glob('*.json'):
            if not ('discovery_' in path.name or 'validation_' in path.name):
                continue
            if family in ('vmem', 'cmem') and not path.name.startswith(family + '_'):
                continue
            data = json.loads(path.read_text())
            assert len(data['records']) == 24
            for row in data['records']:
                assert row['lcc'][1] - row['lcc'][0] == row['cycles']
                assert row.get('full_payload_and_guards_correct', row.get('full_payload_correct', False))
            actual.add((data['size_bytes'], data['window']))
        comparisons[f'BC/Host {family}'] = (expected, actual)
    expected, actual = {tuple(row) for row in legacy['local_dma']}, set()
    for path in (root / '02_local_suite').glob('*/summary.json'):
        for row in json.loads(path.read_text())['records']:
            if row['boundary'] == 'complete':
                assert row['correct']
                actual.add((row['path'], row['size_bytes'], row['window'], row['cores']))
    comparisons['local DMA'] = (expected, actual)
    for space in ('vmem', 'cmem'):
        expected, actual = {tuple(row) for row in legacy['remote'][space]}, set()
        for path in (root / ('04_remote_suite' if space == 'vmem' else '04_cmem_suite')).glob('*/summary.json'):
            name = path.parent.name.rsplit('_', 1)[0].rsplit('_w', 1)[0]
            for row in remote.records(path):
                actual.add((name, row['size_bytes'], row['window']))
        comparisons[f'remote {space}'] = (expected, actual)
    expected, actual = {tuple(row) for row in legacy['register_working_sets']}, set()
    planned = {(space, job['size_kib'] * 1024, job['streams'], job['accumulators'], job['cores']) for job in registers.jobs() for space in job['spaces']}
    assert expected == planned, '循环读取矩阵须完整保留旧版工作集、展开、累加器和 TC 数的组合'
    for path in (root / '11_register_suite').glob('*.validated.json'):
        job = json.loads(path.read_text())['job']
        actual.update((space, job['size_kib'] * 1024, job['streams'], job['accumulators'], job['cores']) for space in job['spaces'])
    comparisons['register working sets'] = (expected, actual)
    for name, (expected, actual) in comparisons.items():
        assert partial or expected <= actual, f'旧版 {name} 缺少周期测量：{sorted(expected - actual)}'
        print('legacy scope', name, f'{len(expected & actual)}/{len(expected)}', 'configurations covered')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/tmp/tpu_latency_numbers'))
    parser.add_argument('--partial', action='store_true', help='只检查已完成组并报告缺口；默认要求全部矩阵完成')
    args = parser.parse_args()
    verify_remote(args.root / 'evidence/04_remote', 'vmem', args.partial)
    verify_remote(args.root / 'evidence/04_cmem', 'cmem', args.partial)
    verify_rtt(args.root / 'evidence/08_rtt', args.partial)
    verify_ring(args.root / '10_hbm_ring_suite', args.partial)
    verify_registers(args.root / '11_register_suite', args.partial)
    verify_legacy_scope(args.root, args.partial)
