"""分离测量 TPU v4 CLD 发射、CRF 排空与 pop-to-use 行为。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np

vld = import_module('01_vld')
cld = import_module('03_cld')
bundle, read, gap = vld.bundle, vld.read, vld.gap

def expected_sum(probe: vld.Probe, count: int) -> np.ndarray:
    expected = np.zeros((8, 128), np.uint32)
    for index in range(count):
        expected += probe.host[(index % 32) * 8:(index % 32 + 1) * 8]
    return expected

def load_only(probe: vld.Probe, counts: list[int]) -> None:
    for count in counts:
        for spacing in ('dense', 'vnop'):
            setup = cld.initialize() + bundle('va0: vimm.8x128.s32 v10, 0') + gap('vnop', 16)
            loads = []
            for index in range(count):
                loads.append(bundle(f'cld: cld.8x128 crf, [cmem:0x{(index % 32) * 8:x}]'))
                if spacing == 'vnop' and index + 1 < count:
                    loads.append(bundle('misc: vnop'))
            body = read(20) + ''.join(loads) + read(21) + bundle('s0: sfence') + read(22)
            # R2 已经采完才排空 CRF；排空等待不计入被测区间，但完整累加检查每个结果。
            for _ in range(count):
                body += bundle('vr0: vpop.8x128 v11, crf') + bundle('va0: vadd.8x128.s32 v10, v11, v10')
            record = probe.run(f'load-{spacing}-n{count}', body, expected_sum(probe, count), setup)
            assert all(record['correct'])

def aged_setup(count: int) -> str:
    setup = cld.initialize()
    setup += ''.join(bundle(f'cld: cld.8x128 crf, [cmem:0x{(index % 32) * 8:x}]') for index in range(count))
    # 远大于现有单条 cld 探针的 67-cycle 完成边界，并由 Probe 的后续 sfence 收束向量发射。
    return setup + gap('vnop', 96)

def aged_pop(probe: vld.Probe, counts: list[int]) -> None:
    destinations = (11, 12, 13, 14)
    for count in counts:
        for slot in ('vr0', 'vr1'):
            for destination_mode in ('same', 'round-robin'):
                registers = [11] * count if destination_mode == 'same' else [destinations[index % len(destinations)] for index in range(count)]
                work = ''.join(bundle(f'{slot}: vpop.8x128 v{register}, crf') for register in registers)
                body = read(20) + work + read(21) + bundle('s0: sfence') + read(22)
                body += gap('vnop', 16) + bundle(f'va0: vmov.8x128 v10, v{registers[-1]}')
                expected = probe.host[((count - 1) % 32) * 8:((count - 1) % 32 + 1) * 8]
                record = probe.run(f'pop-{slot}-{destination_mode}-n{count}', body, expected, aged_setup(count))
                assert all(record['correct'])

def aged_eup_pop(probe: vld.Probe, counts: list[int]) -> None:
    destinations = (11, 12, 13, 14)
    for count in counts:
        if count > 16:
            continue
        setup = bundle('va0: vimm.8x128.s32 v9, 0') + gap('vnop', 16)
        setup += bundle('va0: vpush.8x128 erf, v9') * count + gap('vnop', 96)
        registers = [destinations[index % len(destinations)] for index in range(count)]
        work = ''.join(bundle(f'vr0: vpop.8x128 v{register}, erf') for register in registers)
        body = read(20) + work + read(21) + bundle('s0: sfence') + read(22)
        body += gap('vnop', 16) + bundle(f'va0: vmov.8x128 v10, v{registers[-1]}')
        expected = np.zeros((8, 128), np.uint32)
        record = probe.run(f'pop-erf-n{count}', body, expected, setup)
        assert all(record['correct'])

def pop_use(probe: vld.Probe) -> None:
    independent = bundle('vld: vld.8x128 v9, [vmem:0x0]') + gap('vnop', 16)
    for dependency in ('independent', 'dependent'):
        source = 'v9' if dependency == 'independent' else 'v11'
        body = read(20) + bundle('vr0: vpop.8x128 v11, crf') + bundle(f'va0: vadd.8x128.s32 v10, 1, {source}')
        body += read(21) + bundle('s0: sfence') + read(22)
        expected = probe.host[:8] + np.uint32(1)
        record = probe.run(f'pop-use-{dependency}', body, expected, aged_setup(1) + independent)
        assert all(record['correct'])

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/03_cld_service'))
    parser.add_argument('--repeats', type=int, default=24)
    parser.add_argument('--counts', type=lambda value: [int(item) for item in value.split(',')], default=[1, 2, 4, 8, 16, 24, 28])
    parser.add_argument('--group', choices=('load', 'pop', 'eup', 'use', 'safe'), default='safe')
    args = parser.parse_args()
    if min(args.counts) < 1 or max(args.counts) > 28 or args.repeats < 1:
        parser.error('count 须为 1..28，repeats 须为正')
    probe = vld.Probe(args.output, args.repeats, ('load-dense-n28', 'pop-vr0-round-robin-n28', 'pop-erf-n16'))
    if args.group in ('load', 'safe'):
        load_only(probe, args.counts)
    if args.group in ('pop', 'safe'):
        aged_pop(probe, args.counts)
    if args.group in ('eup', 'safe'):
        aged_eup_pop(probe, args.counts)
    if args.group in ('use', 'safe'):
        pop_use(probe)
    probe.baseline()
    (args.output / 'summary.json').write_text(json.dumps({'environment': probe.environment, 'records': probe.records, 'baseline_after': True}, indent=2) + '\n')

if __name__ == '__main__':
    main()
