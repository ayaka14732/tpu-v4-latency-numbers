"""仅在 CPU 上重放归档的 BC／Host paired 计数、校准公式与独立尺寸残差。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np

local = import_module('02_local_model')
refine = import_module('06_host_refine')

def records(path: Path) -> dict:
    data = json.loads(path.read_text())
    assert len(data['records']) == 24, path
    for row in data['records']:
        assert row['lcc'][1] - row['lcc'][0] == row['cycles'], path
        if 'full_payload_and_guards_correct' in row:
            assert row['full_payload_and_guards_correct'] and data['cleanup_clean'], path
        else:
            assert row['full_payload_correct'] and row['pinned_host_verified'] and data['baseline_after'] and data['poison_input_unchanged'], path
    return {'size_bytes': data['size_bytes'], 'cycles': [row['cycles'] for row in data['records']]}

def verify(root: Path) -> None:
    for name in ('05_bc', '06_host', '06_host_refine'):
        folder = root / name
        windows = (4, 16) if name == '06_host_refine' else (1, 4, 16)
        assert {path.name for path in folder.glob('w*.validated.json')} == {f'w{window}.validated.json' for window in windows}
        rows = ['| W | raw 周期中心公式，K=S/1024 | 验证中位数最大误差 | 验证全样本残差 |', '| ---: | --- | ---: | --- |']
        total = 0
        for path in sorted(folder.glob('w*.validated.json')):
            model = json.loads(path.read_text())
            frozen = json.loads(path.with_name(path.name.replace('.validated.', '.model.')).read_text())
            assert {key: value for key, value in model.items() if key != 'validation'} == frozen
            window = model['window']
            for phase in ('discovery', 'validation'):
                files = sorted(folder.glob(f'{phase}_w{window}_s*.json'), key=lambda p: int(p.stem.split('_s')[-1]))
                group = [records(p) for p in files]
                errors = refine.errors(group, model) if model['kind'] == 'empirical_continuous_hinge' else local.errors(group, model)
                assert set(errors) == set(model[phase]), path
                for key, value in errors.items():
                    np.testing.assert_allclose(value, model[phase][key], rtol=1e-12, atol=1e-8, err_msg=f'{path}: {phase}: {key}')
                total += errors['samples']
            formula = f'{model["intercept_cycles"]:.3f} + {model["cycles_per_kib"]:.5f}K'
            if model['kind'] == 'empirical_continuous_hinge':
                formula += f' + {model["extra_cycles_per_kib"]:.5f}·max(K−{model["knot_kib"]},0)'
            check = model['validation']
            rows.append(f'| {window} | {formula} | {check["max_median_error_cycles"]:.2f} | [{check["min_residual_cycles"]:.2f},{check["max_residual_cycles"]:.2f}] |')
        assert total == {'05_bc': 936, '06_host': 792, '06_host_refine': 960}[name]
        (folder / 'formulas.md').write_text('\n'.join(rows) + '\n')
        print(name, total, 'raw samples and frozen model residuals verified')
    for gap in (0, 1, 4, 16, 64):
        data = records(root / '05_bc' / f'gap{gap}.json')
        assert len(data['cycles']) == 24 and set(data['cycles']) == {gap + 1}
        data = records(root / '06_host' / f'gap{gap}.json')
        assert len(data['cycles']) == 24 and set(data['cycles']) == {gap + 21}
    for size in (4096, 1048576):
        data = records(root / '06_host' / f'issue_s{size}.json')
        assert len(data['cycles']) == 24 and set(data['cycles']) == {2}
    print('BC paired calibration: 120 exact samples; Host SMEM calibration: 120; issue-only control: 48')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence'))
    verify(parser.parse_args().evidence)
