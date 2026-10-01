"""仅在 CPU 上核对 Host Magic Queue 的设备周期、空窗口、完整数据记录和冻结模型。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np

local = import_module('02_local_model')

def records(path: Path, empty: bool = False) -> dict:
    data = json.loads(path.read_text())
    assert data['protocol'] == 'host-magic-bc-lcc-envelope-v2'
    assert data['includes_host_bc_marker_handshake'] and data['counter_words_read_after_end'] and data['cleanup_clean']
    assert data['empty_window'] == empty
    assert data['working_set_bytes'] == (64 * 1024**2 // data['size_bytes']) * data['size_bytes']
    assert len(data['records']) == 24
    for index, row in enumerate(data['records']):
        assert row['full_payload_and_guards_correct'] and row['sequence'] == 2 * (index + 3)
        assert row['lcc'][1] - row['lcc'][0] == row['cycles'] > 0
    return {'size_bytes': data['size_bytes'], 'cycles': [row['cycles'] for row in data['records']]}

def main(root: Path) -> None:
    rows = ['| W | raw 周期中心公式，K=S/1024 | 验证中位数最大误差 | 最大相对误差 | 验证全样本残差 |', '| ---: | --- | ---: | ---: | --- |']
    controls = ['| W | 空窗口中位数 | 空窗口范围 |', '| ---: | ---: | --- |']
    total = 0
    for window in (1, 4, 16, 64):
        model = json.loads((root / f'w{window}.validated.json').read_text())
        frozen = json.loads((root / f'w{window}.model.json').read_text())
        assert {key: value for key, value in model.items() if key != 'validation'} == frozen
        control = records(root / f'control_w{window}_s4096.json', empty=True)['cycles']
        controls.append(f'| {window} | {np.median(control):.0f} | [{min(control)}, {max(control)}] |')
        for phase in ('discovery', 'validation'):
            group = [records(path) for path in sorted(root.glob(f'{phase}_w{window}_s*.json'))]
            errors = local.errors(group, model)
            for key, value in errors.items():
                np.testing.assert_allclose(value, model[phase][key], rtol=1e-12, atol=1e-8)
            total += errors['samples']
        relative = []
        for record in group:
            prediction = model['intercept_cycles'] + model['cycles_per_kib'] * record['size_bytes'] / 1024
            median = float(np.median(record['cycles']))
            relative.append(abs(prediction - median) / median)
        check = model['validation']
        formula = f'{model["intercept_cycles"]:.3f} + {model["cycles_per_kib"]:.5f}K'
        rows.append(f'| {window} | {formula} | {check["max_median_error_cycles"]:.2f} | {max(relative):.2%} | [{check["min_residual_cycles"]:.2f}, {check["max_residual_cycles"]:.2f}] |')
    assert total == 1008
    note = '同一 BC 的 paired LCC 包围完整 Host 发起／callback／端点握手。空窗口没有从公式扣除；W=4/16/64 的单条仿射模型偏差较大，不能当作精确预测。'
    (root / 'formulas.md').write_text(note + '\n\n' + '\n'.join(rows) + '\n\n' + '\n'.join(controls) + '\n')
    print(total, 'DMA samples and 96 empty-window samples verified')

def verify_refined(root: Path) -> None:
    refine = import_module('06_host_refine')
    total = 0
    for window in (4, 16, 64):
        model = json.loads((root / f'w{window}.validated.json').read_text())
        frozen = json.loads((root / f'w{window}.model.json').read_text())
        assert {key: value for key, value in model.items() if key != 'validation'} == frozen
        sizes = {}
        for phase, seed in (('discovery', 531), ('validation', 532)):
            paths = sorted(root.glob(f'{phase}_w{window}_s*.json'))
            group = [records(path) for path in paths]
            assert all(json.loads(path.read_text())['seed'] == seed for path in paths)
            sizes[phase] = {row['size_bytes'] for row in group}
            errors = refine.errors(group, model)
            for key, value in errors.items():
                np.testing.assert_allclose(value, model[phase][key], rtol=1e-12, atol=1e-8)
            total += errors['samples']
        assert not sizes['discovery'] & sizes['validation']
    assert total == 1416
    print(total, 'Magic Queue refined samples, frozen parameters and independent residuals verified')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/09_magic_host'))
    parser.add_argument('--refined', type=Path, help='同时核对新一轮连续分段模型的本机记录目录')
    args = parser.parse_args()
    main(args.evidence)
    if args.refined:
        verify_refined(args.refined)
