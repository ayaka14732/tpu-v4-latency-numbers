"""测量 Megacore Shared CMEM 经 cld/CRF/vpop 到 TC VREG 的周期与流水成本。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np

vld = import_module('01_vld')
local = import_module('02_local_dma')
bundle, read, gap = vld.bundle, vld.read, vld.gap

def initialize() -> str:
    return local.copy('[cmem:s24]', '[vmem:s24]', 256)

def consumers(probe: vld.Probe) -> None:
    for distance in (0, 1, 2, 4, 8, 16, 32, 64):
        # cld 先发出；pop 后的 consumer 检查数据就绪，避免把 CRF 等待与 pop-use 混为一谈。
        load = bundle('cld: cld.8x128 crf, [cmem:0x0]')
        pop = 'vr0: vpop.8x128 v11, crf'
        consumer = 'va0: vadd.8x128.s32 v10, 1, v11'
        work = load + (bundle(pop + ' ; ' + consumer) if distance == 0 else bundle(pop) + gap('vnop', distance - 1) + bundle(consumer))
        body = read(20) + work + read(21) + bundle('s0: sfence') + read(22)
        expected = np.full((8, 128), vld.SENTINEL + 1, np.uint32) if distance == 0 else probe.host[:8] + np.uint32(1)
        record = probe.run(f'pop-use-{distance}', body, expected, initialize())
        assert all(record['correct'])
    for distance in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        body = read(20) + bundle('cld: cld.8x128 crf, [cmem:0x0]') + gap('vnop', distance - 1)
        body += bundle('vr0: vpop.8x128 v10, crf') + read(21) + bundle('s0: sfence') + read(22)
        record = probe.run(f'cld-pop-{distance}', body, probe.host[:8], initialize())
        assert all(record['correct'])

def streams(probe: vld.Probe, depths: list[int], counts: list[int], schedule: str = 'split') -> None:
    for count in counts:
        for depth in depths:
            if depth > count:
                continue
            setup = initialize() + bundle('va0: vimm.8x128.s32 v10, 0') + gap('vnop', 16)
            if depth == 0:
                work = ''.join(bundle(f'cld: cld.8x128 crf, [cmem:0x{(i % 32) * 8:x}]') + bundle('vr0: vpop.8x128 v11, crf') + bundle('va0: vadd.8x128.s32 v10, v11, v10') for i in range(count))
            else:
                work = ''.join(bundle(f'cld: cld.8x128 crf, [cmem:0x{i * 8:x}]') for i in range(depth))
                for index in range(count):
                    instruction = 'vr0: vpop.8x128 v11, crf'
                    if index + depth < count:
                        instruction += f' ; cld: cld.8x128 crf, [cmem:0x{((index + depth) % 32) * 8:x}]'
                    if schedule == 'fused' and index > 0:
                        instruction += ' ; va0: vadd.8x128.s32 v10, v11, v10'
                    work += bundle(instruction)
                    if schedule == 'split':
                        work += bundle('va0: vadd.8x128.s32 v10, v11, v10')
                if schedule == 'fused':
                    # 同 bundle 的 add 读到上一次 pop 的值，最后单独消费最后一个结果。
                    work += bundle('va0: vadd.8x128.s32 v10, v11, v10')
            body = read(20) + work + read(21) + bundle('s0: sfence') + read(22)
            expected = np.zeros((8, 128), np.uint32)
            for index in range(count):
                expected += probe.host[(index % 32) * 8:(index % 32 + 1) * 8]
            prefix = 'stream' if schedule == 'split' else 'fused'
            record = probe.run(f'{prefix}-n{count}-d{depth}', body, expected, setup)
            assert all(record['correct'])

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/03_cld'))
    parser.add_argument('--repeats', type=int, default=24)
    parser.add_argument('--depths', type=lambda value: [int(item) for item in value.split(',')], default=[0, 1, 2, 4])
    parser.add_argument('--counts', type=lambda value: [int(item) for item in value.split(',')], default=[1, 4, 16, 64, 128])
    parser.add_argument('--group', choices=('consumers', 'streams', 'all'), default='all')
    parser.add_argument('--schedule', choices=('split', 'fused'), default='split')
    args = parser.parse_args()
    if any(not 0 <= depth <= 28 for depth in args.depths) or min(args.counts) < 1 or args.repeats < 1:
        parser.error('depth 须为 0..28，count 与 repeats 须为正')
    probe = vld.Probe(args.output, args.repeats, ('stream-n128-d28', 'stream-n128-d4', 'fused-n128-d28', 'pop-use-1'))
    if args.group in ('consumers', 'all'):
        consumers(probe)
    if args.group in ('streams', 'all'):
        streams(probe, args.depths, args.counts, args.schedule)
    probe.baseline()
    (args.output / 'summary.json').write_text(json.dumps({'environment': probe.environment, 'records': probe.records, 'baseline_after': True}, indent=2) + '\n')

if __name__ == '__main__':
    main()
