"""生成同条件路径对比和公式表；仅重放现有周期记录，不重新采样或拟合。"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import import_module
import json
from pathlib import Path
import re

import numpy as np

local = import_module('02_local_model')
remote = import_module('04_remote_suite')
registers = import_module('11_register_suite')
phase_model = import_module('11_register_phase')
ring = import_module('10_hbm_ring_suite')
dma = import_module('05_06_dma_model')
magic = import_module('09_magic_model')
refine = import_module('06_host_refine')
cld = import_module('03_cld_suite')

LABELS = {
    'hbm_vmem': 'HBM → TC VMEM',
    'vmem_hbm': 'TC VMEM → HBM',
    'hbm_cmem': 'HBM → Megacore Shared CMEM',
    'cmem_hbm': 'Megacore Shared CMEM → HBM',
    'cmem_vmem': 'Megacore Shared CMEM → TC VMEM',
    'vmem_cmem': 'TC VMEM → Megacore Shared CMEM',
}
TABLES = {
    '02_local': '本地 DMA · 完整条件',
    '04_remote': '远程 TC VMEM · 全部拓扑与窗口',
    '04_cmem': '远程 Megacore Shared CMEM · 全部拓扑与窗口',
    '05_bc': 'HBM → BC 私有 BMEM',
    '06_host': 'TC 发起 → pinned Host · 初始模型',
    '06_host_refine': 'TC 发起 → pinned Host · 分段模型',
    '07_bc_pull': 'TC VMEM／Megacore Shared CMEM → BC 私有 BMEM',
    '08_rtt': '完整 payload RTT',
    '09_magic_host': 'Host Magic Queue · 初始模型',
    '09_magic_refine': 'Host Magic Queue · 分段模型',
    '10_hbm_ring': 'HBM 环形工作集 · 完整条件',
    '11_register_ring': '循环读取 · 逐配置中心公式',
}

def summarize(values: list[int], **identity: object) -> dict:
    assert len(values) in (12, 24)
    return dict(**identity, count=len(values), median=float(np.median(values)), minimum=min(values), maximum=max(values))

def local_data(root: Path) -> list[dict]:
    directory = root / '02_local_suite'
    models = json.loads((directory / 'validated_models.json').read_text())['models']
    phases = {phase: local.load_phase(directory, phase) for phase in ('discovery', 'validation')}
    result = []
    for key, model in models.items():
        rows = []
        for phase, records in phases.items():
            for row in records:
                if local.model_key(row) == key and row['size_bytes'] >= 4096:
                    rows.append(summarize(row['cycles'], phase=phase, size=row['size_bytes'] // 1024))
        result.append(dict(path=model['path'], label=LABELS[model['path']], model=model, records=rows))
    assert len(result) == 54
    return result

def check_model(group: list[dict], model: dict, phase: str, measure: Callable[[list[dict], dict], dict] = local.errors) -> None:
    errors = measure(group, model)
    assert errors.keys() == model[phase].keys()
    for key, value in errors.items():
        np.testing.assert_allclose(value, model[phase][key], rtol=1e-12, atol=1e-8)

def frozen_model(path: Path) -> dict:
    model = json.loads(path.read_text())
    frozen = json.loads(path.with_name(path.name.replace('.validated.', '.model.')).read_text())
    assert {key: value for key, value in model.items() if key != 'validation'} == frozen
    return model

def ring_data(root: Path) -> list[dict]:
    directory = root / '10_hbm_ring_suite'
    result = []
    for cores in (1, 2):
        for window in (1, 4, 8):
            name = f'c{cores}_w{window}'
            models = json.loads((directory / f'{name}.validated.json').read_text())
            frozen = json.loads((directory / f'{name}.model.json').read_text())
            assert [{k: v for k, v in model.items() if k != 'validation'} for model in models] == frozen
            phases = {phase: ring.records(directory / f'{name}_{phase}' / 'summary.json') for phase in ('discovery', 'validation')}
            for model in models:
                rows = []
                for phase, records in phases.items():
                    selected = [r for r in records if r['path'] == model['path'] and r['core'] == model['core']]
                    check_model(selected, model, phase)
                    rows.extend(summarize(r['cycles'], phase=phase, size=r['size_bytes'] // 1024) for r in selected)
                result.append(dict(path=model['path'], label=LABELS[model['path']], model=dict(model, cores=cores, window=window), records=rows))
    assert len(result) == 36
    return result

def remote_data(root: Path, rtt: bool = False) -> list[dict]:
    result = []
    folders = [('vmem', '08_rtt')] if rtt else [('vmem', '04_remote'), ('cmem', '04_cmem')]
    for space, folder in folders:
        directory = root / 'evidence' / folder
        for path in sorted(directory.glob('*.validated.json')):
            name = path.name.removesuffix('.validated.json')
            document = json.loads(path.read_text())
            frozen = json.loads((directory / f'{name}.model.json').read_text())
            protocol = 'remote_lcc_v2_payload_rtt' if rtt else remote.PROTOCOL
            phases = {phase: remote.records(directory / f'{name}_{phase}.json', protocol) for phase in ('discovery', 'validation')}
            for source, target in document['flows']:
                model = document['models'][str(source)]
                assert {k: v for k, v in model.items() if k != 'validation'} == frozen['models'][str(source)]
                rows, storage = [], set()
                for phase, records in phases.items():
                    selected = remote.source_records(records, source)
                    check_model(selected, model, phase)
                    for row in selected:
                        storage.add(row['audit'].get('counter_storage', 'registers'))
                        rows.append(summarize(row['cycles'], phase=phase, size=row['size_bytes'] // 1024))
                assert len(storage) == 1
                result.append(dict(
                    case=name if rtt else name.rsplit('_w', 1)[0],
                    space=space,
                    source=source,
                    target=target,
                    cores=document['cores'],
                    window=document['window'],
                    storage=storage.pop(),
                    model=model,
                    records=rows,
                ))
    assert len(result) == (16 if rtt else 138)
    return result

def bc_pull_record(path: Path) -> dict:
    document = json.loads(path.read_text())
    assert document['residents_stopped'] and document['hbm_freed'] and document['observer_removed']
    assert document['protocol'] == 'bc-pull-paired-lcc-v1' and document['libtpu'] == '0.0.46'
    assert len(document['records']) == 24
    for row in document['records']:
        assert row['full_payload_and_guards_correct'] and row['tc_source_released']
        assert row['lcc'][1] - row['lcc'][0] == row['cycles']
    return dict(size_bytes=document['size_bytes'], cycles=[row['cycles'] for row in document['records']])

def scalar_dma_data(directory: Path, prefix: str, window: int, reader: Callable[[Path], dict]) -> dict:
    model = frozen_model(directory / f'{prefix}w{window}.validated.json')
    rows = []
    for phase in ('discovery', 'validation'):
        group = [reader(path) for path in sorted(directory.glob(f'{prefix}{phase}_w{window}_s*.json'))]
        check_model(group, model, phase, refine.errors if model['kind'] == 'empirical_continuous_hinge' else local.errors)
        rows.extend(summarize(r['cycles'], phase=phase, size=r['size_bytes'] // 1024) for r in group)
    return dict(window=window, model=model, records=rows)

def bmem_data(root: Path) -> list[dict]:
    result = []
    for space, label in [('hbm', 'HBM'), ('vmem', 'TC VMEM'), ('cmem', 'Megacore Shared CMEM')]:
        directory = root / 'evidence' / ('05_bc' if space == 'hbm' else '07_bc_pull')
        for window in (1, 4, 16):
            group = scalar_dma_data(directory, '' if space == 'hbm' else space + '_', window, dma.records if space == 'hbm' else bc_pull_record)
            result.append(dict(group, space=space, label=label))
    assert sum(r['count'] for g in result for r in g['records']) == 2328
    return result

def host_data(root: Path) -> list[dict]:
    result = []
    for initiator, folder, refined, windows, reader in [
        ('tc', '06_host', '06_host_refine', (1, 4, 16), dma.records),
        ('magic', '09_magic_host', '09_magic_refine', (1, 4, 16, 64), magic.records),
    ]:
        for cohort in ('initial', 'refined'):
            for window in windows:
                if cohort == 'refined' and window == 1:
                    continue
                group = scalar_dma_data(root / 'evidence' / (folder if cohort == 'initial' else refined), '', window, reader)
                result.append(dict(group, initiator=initiator, cohort=cohort))
    assert len(result) == 12
    assert sum(r['count'] for g in result for r in g['records']) == 4176
    return result

def register_data(root: Path) -> list[dict]:
    directory = root / '11_register_suite'
    assert json.loads((directory / 'phase_candidate.json').read_text()) == phase_model.specification()
    result = []
    for job in registers.jobs():
        name = f'{job["family"]}_s{job["size_kib"]}_u{job["streams"]}_a{job["accumulators"]}_c{job["cores"]}'
        rows = []
        for phase in ('discovery', 'validation'):
            path = directory / f'{name}_{phase}' / 'summary.json'
            registers.records(path)
            for row in json.loads(path.read_text())['records']:
                slope, intercept = phase_model.model.candidate(row['space'], job['streams'], job['accumulators'])
                for core in range(job['cores']):
                    values = []
                    for sample in row['records']:
                        cycles = sample['cycles'][core]
                        parity = sample['lcc'][core][0] & 1 if row['space'] == 'cmem' else 0
                        assert cycles == slope * row['iterations'] + intercept + parity
                        values.append(cycles)
                    rows.append(summarize(values, phase=phase, iterations=row['iterations'], space=row['space'], core=core, slope=slope, intercept=intercept))
        result.append(dict(job, name=name, records=rows))
    assert len(result) == 52
    assert sum(r['count'] for g in result for r in g['records']) == 47232
    return result

def instruction_data(root: Path) -> list[dict]:
    rows = []
    for line in (root / '01_vld/results.jsonl').read_text().splitlines():
        record = json.loads(line)
        if match := re.fullmatch(r'stream-vld-(\d+)', record['name']):
            count = int(match[1])
            assert all(record['correct']) and all(words == 1024 for words in record['matching_words'])
            values = []
            for lcc, gaps in zip(record['lcc'], record['gaps_from_begin'], strict=True):
                assert gaps == [lcc[1] - lcc[0], lcc[2] - lcc[0]] and gaps[1] == count + 13
                values.append(gaps[1])
            rows.append(summarize(values, phase='probe', size=4 * count))
    result = [dict(space='vmem', schedule='split', depth=0, label='TC VMEM · vld', formula='N + 13', records=rows)]
    groups = {}
    for job in cld.jobs():
        path = root / 'evidence/03_cld' / f'{job["name"]}.json'
        cld.verify_job(path, job)
        for record in json.loads(path.read_text())['records']:
            if match := re.fullmatch(r'(?:stream|fused)-n(\d+)-d(\d+)', record['name']):
                count, depth = map(int, match.groups())
                phase = 'validation' if 'holdout' in job['name'] else 'discovery'
                groups.setdefault((job['schedule'], depth), []).append(summarize([gaps[1] for gaps in record['gaps_from_begin']], phase=phase, size=4 * count))
    for (schedule, depth), rows in sorted(groups.items()):
        formula = '56N + 11' if depth == 0 else f'{max(67, 2 * depth + 14)} + {max(54, 2 * depth)}⌊(N−1)/{depth}⌋ + 2((N−1) mod {depth})'
        result.append(dict(space='cmem', schedule=schedule, depth=depth, label='Megacore Shared CMEM · ' + ('串行累加' if depth == 0 else f'D={depth}'), formula=formula, records=rows))
    assert len(result) == 20
    return result

def formula_tables(here: Path) -> list[dict]:
    result = []
    for name, title in TABLES.items():
        path = here.parent / 'results' / f'{name}.md'
        contents = path.read_text()
        note = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', contents.split('\n\n')[0]).replace('`', '')
        blocks = re.findall(r'(?:^\|[^\n]*\n?)+', contents, flags=re.MULTILINE)
        for index, block in enumerate(blocks):
            cells = [[cell.strip().replace('`', '') for cell in line.strip('|').split('|')] for line in block.splitlines()]
            assert all(len(row) == len(cells[0]) for row in cells)
            suffix = ' · 空窗口' if index else ''
            result.append(dict(name=name + ('_control' if index else ''), title=title + suffix, note=note, columns=cells[0], rows=cells[2:]))
    return result

def render(root: Path, output: Path) -> None:
    here = Path(__file__).resolve().parent
    data = dict(
        local=local_data(root),
        ring=ring_data(root),
        remote=remote_data(root),
        rtt=remote_data(root, rtt=True),
        bmem=bmem_data(root),
        host=host_data(root),
        registers=register_data(root),
        instructions=instruction_data(root),
        tables=formula_tables(here),
    )
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    vendor = here.parent / 'vendor' / 'chart.umd.min.js'
    chart_js = vendor.read_text().split('\n//# sourceMappingURL=', 1)[0]
    page = (here / '13_visualize.html').read_text().replace('__CHART_JS__', chart_js).replace('__DATA__', encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    print(f'{output}: {output.stat().st_size} bytes; groups={ {key: len(value) for key, value in data.items()} }')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/tmp/tpu_latency_numbers'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/visualization/index.html'))
    args = parser.parse_args()
    render(args.root, args.output)
