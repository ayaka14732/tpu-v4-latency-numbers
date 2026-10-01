"""在 CPU 上重放 BC pull 的原始周期、源释放、清理记录和冻结公式。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np

local = import_module('02_local_model')

def main(root: Path) -> None:
    expected = {f'{space}_w{window}.validated.json' for space in ('vmem', 'cmem') for window in (1, 4, 16)}
    assert {path.name for path in root.glob('*_w*.validated.json')} == expected
    rows = ['| 来源 | W | raw 周期中心公式，K=S/1024 | 验证中位数最大误差 | 验证全样本残差 |', '| --- | ---: | --- | ---: | --- |']
    samples = 0
    for path in sorted(root.glob('*_w*.validated.json')):
        model = json.loads(path.read_text())
        frozen = json.loads(path.with_name(path.name.replace('.validated.', '.model.')).read_text())
        assert {key: value for key, value in model.items() if key != 'validation'} == frozen
        for phase in ('discovery', 'validation'):
            files = sorted(root.glob(f'{model["space"]}_{phase}_w{model["window"]}_s*.json'), key=lambda p: int(p.stem.split('_s')[-1]))
            group = []
            for file in files:
                data = json.loads(file.read_text())
                assert len(data['records']) == 24
                assert data['residents_stopped'] and data['hbm_freed'] and data['observer_removed']
                for row in data['records']:
                    assert row['full_payload_and_guards_correct'] and row['tc_source_released']
                    assert row['lcc'][1] - row['lcc'][0] == row['cycles']
                group.append({'size_bytes': data['size_bytes'], 'cycles': [row['cycles'] for row in data['records']]})
            errors = local.errors(group, model)
            for key, value in errors.items():
                np.testing.assert_allclose(value, model[phase][key], rtol=1e-12, atol=1e-8)
            samples += errors['samples']
        check = model['validation']
        formula = f'{model["intercept_cycles"]:.3f} + {model["cycles_per_kib"]:.5f}K'
        rows.append(f'| {model["space"]} | {model["window"]} | {formula} | {check["max_median_error_cycles"]:.2f} | [{check["min_residual_cycles"]:.2f},{check["max_residual_cycles"]:.2f}] |')
    assert samples == 1392
    (root / 'formulas.md').write_text('\n'.join(rows) + '\n')
    print(samples, 'BC paired LCC samples, source releases and frozen model residuals verified')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/07_bc_pull'))
    main(parser.parse_args().evidence)
