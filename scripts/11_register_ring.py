"""保留工作集、展开数、累加器数与单／双 TC 的循环读取实验，直接测量 paired LCC。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import re

local = import_module('02_local_dma')
jax, np = local.jax, local.np

def fragment(space: str, rows: int, streams: int, accumulators: int, iterations: int, cores: int, mapping: dict[int, int]) -> str:
    b = local.bundle
    text = f'.target {local.TARGET}\n' + b('s0: sfence')
    text += b('s1: sld s28, [smem:0x1]') + b(f's0: smul.u32 s28, {rows}, s28')
    if space == 'cmem':
        text += b('s0: simm.s32 s23, 0') + local.copy('[cmem:s28]', '[vmem:s23]', rows)
    for accumulator in range(accumulators):
        text += b(f'va0: vimm.8x128.s32 v{accumulator}, 0')
    if accumulators == 32:
        text += b(f'vst: vst.8x128 [vmem:0x{rows + 31 * 8:x}], v31')
    if cores == 2:
        text += b('s1: sld s20, [smem:0x0]') + b('s1: sld s21, [smem:0x1]')
        text += b('s0: sand.u32 s20, 0xfff, s20 ; s1: ssub.s32 s21, 1, s21')
        text += b('s0: sshll.u32 s20, s20, 0x12 ; s1: sshll.u32 s21, s21, 0xe')
        text += b('s0: sor.u32 s20, s20, s21') + b('s0: sor.u32 s20, 0x802d, s20')
        text += b('misc: vsyncadd.remote.s32 [sflag:s20], 1') + b('s0: sfence ; misc: vwait.ge [sflag:45], 1')
        text += b('misc: vsyncadd.s32 [sflag:45], -1')
    text += b(f's0: simm.s32 s23, 0 ; s1: simm.s32 s24, {iterations}') + b('s0: sfence') + local.read(20)
    text += 'loop:\n'
    text += b('s0: sadd.s32 s29, s23, s28') if space == 'cmem' else b()
    depth = min(streams, 16)
    if space == 'cmem':
        text += ''.join(b(f'cld: cld.8x128 crf, [cmem:s29 + 0x{index * 8:x}]') for index in range(depth))
    for index in range(streams):
        accumulator = index % accumulators
        temporary = min(accumulators, 31)
        if accumulator == 31:
            # 32 个独立累加器占满 TC VREG：把 v30 临时写出，为最后一次读取腾位置。
            text += b(f'vst: vst.8x128 [vmem:0x{rows + 30 * 8:x}], v30')
            text += b(f'vld: vld.8x128 v31, [vmem:0x{rows + 31 * 8:x}]')
            temporary = 30
        if space == 'vmem':
            text += b(f'vld: vld.8x128 v{temporary}, [vmem:s23 + 0x{index * 8:x}]')
        else:
            pop = f'vr0: vpop.8x128 v{temporary}, crf'
            if index + depth < streams:
                pop += f' ; cld: cld.8x128 crf, [cmem:s29 + 0x{(index + depth) * 8:x}]'
            text += b(pop)
        text += b(f'va1: vadd.8x128.f32 v{accumulator}, v{temporary}, v{accumulator}')
        if accumulator == 31:
            text += b(f'vst: vst.8x128 [vmem:0x{rows + 31 * 8:x}], v31')
            text += b(f'vld: vld.8x128 v30, [vmem:0x{rows + 30 * 8:x}]')
    text += b(f's0: sadd.s32 s23, {8 * streams}, s23 ; s1: sadd.s32 s24, -1, s24')
    text += b(f's0: sand.u32 s23, 0x{rows - 1:x}, s23 ; s1: sne.s32 p14, s24, 0')
    text += b('s0: @p14 sbr.rel loop') + b() + local.read(21) + b('s0: sfence') + local.read(22)
    for accumulator in range(min(accumulators, 31)):
        text += b(f'vst: vst.8x128 [vmem:0x{rows + accumulator * 8:x}], v{accumulator}')
    text += b('s0: sfence')
    for tile, register in enumerate((20, 21, 22, 25, 26, 27)):
        text += b(f'va0: vmov.8x128 v12, s{register}') + b('misc: vnop') * 8
        text += b(f'vst: vst.8x128 [vmem:0x{rows + accumulators * 8 + tile * 8:x}], v12')
    text += b('s0: sfence')
    return re.sub(r'(?<![\w.])s(\d+)\b(?!:)', lambda match: f's{mapping[int(match[1])]}', text)

def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    device = jax.local_devices()[0]
    assert device.device_kind == 'TPU v4' and local.version('libtpu') == '0.0.49'
    rows, accumulators = args.size // 512, min(args.streams, args.accumulators or args.streams)
    shape = (args.cores, rows + accumulators * 8 + 48, 128)
    host = np.full(shape, 0xDEADBEEF, np.uint32)
    groups = rows // (args.streams * 8)
    source = np.random.default_rng(args.seed).integers(-8, 9, (args.cores, groups, args.streams, 8, 128), dtype=np.int16).astype(np.float32)
    host[:, :rows] = source.reshape(args.cores, rows, 128).view(np.uint32)
    host_input = host[0] if args.cores == 1 else host
    compiled, x, raw, key, marker_pc, listing, _, _, mapping = local.compile_carrier(rows, host_input, device, args.cores)
    program = local.parse_assembly(listing)
    pc = next(pc for pc, block in enumerate(program.bundles) if pc > marker_pc and any(item.mnemonic == 'dma.simple' and item.operands[0].startswith('[hbm:') for item in block.instructions))
    (args.output / 'carrier.tpuasm').write_text(listing)
    records = []
    for space in args.spaces:
        for count in args.iterations:
            name = f'{space}_n{count}'
            probe = fragment(space, rows, args.streams, accumulators, count, args.cores, mapping)
            patched = local.insert_executable_bundles(raw, {key: [local.BundleInsertion(pc, probe)]})
            (_, _, image), = local.executable_programs(patched)
            (args.output / f'{name}.tpuasm').write_text(probe)
            (args.output / f'{name}.full.tpuasm').write_text(local.format_assembly(image, target=local.TARGET))
            function = local.load_executable(patched, compiled)
            cycles, tail = divmod(count, groups)
            expected = cycles * source.sum(axis=1, dtype=np.float64) + source[:, :tail].sum(axis=1, dtype=np.float64)
            expected = expected.reshape(args.cores, -1, accumulators, 8, 128).sum(axis=1).astype(np.float32)
            samples = []
            for repeat in range(args.repeats + 2):
                result = np.asarray(function(x)).reshape(shape)
                np.testing.assert_array_equal(result[:, :rows], host[:, :rows])
                np.testing.assert_array_equal(result[:, rows:rows + accumulators * 8].reshape(expected.shape).view(np.float32), expected)
                tiles = result[:, rows + accumulators * 8:].reshape(args.cores, 6, 8, 128)
                np.testing.assert_array_equal(tiles, np.broadcast_to(tiles[:, :, :1, :1], tiles.shape))
                halves = tiles[:, :, 0, 0].astype(np.uint64)
                lcc = halves[:, :3] | (halves[:, 3:] << np.uint64(32))
                if repeat >= 2:
                    samples.append(dict(lcc=lcc.tolist(), cycles=(lcc[:, 2] - lcc[:, 0]).tolist(), source_and_all_accumulators_correct=True))
            record = dict(space=space, iterations=count, streams=args.streams, accumulators=accumulators, size_bytes=args.size, cores=args.cores, records=samples)
            records.append(record)
            print(name, 'cycles', np.unique([sample['cycles'] for sample in samples], axis=0).tolist(), flush=True)
    baseline = host.copy()
    baseline[:, :8] ^= np.uint32(0x13579BDF)
    np.testing.assert_array_equal(np.asarray(compiled(x)).reshape(shape), baseline)
    data = dict(
        protocol='register_cyclic_lcc_v1',
        libtpu=local.version('libtpu'),
        jax=jax.__version__,
        device=str(device),
        seed=args.seed,
        baseline_after=True,
        cmem_prefetch_depth=min(args.streams, 16),
        explicit_spill=accumulators == 32,
        records=records,
    )
    (args.output / 'summary.json').write_text(json.dumps(data, indent=2) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(65536, 262144, 1048576, 4194304), default=65536)
    parser.add_argument('--streams', type=int, choices=(1, 4, 8, 16, 32, 64, 128), default=4)
    parser.add_argument('--accumulators', type=int)
    parser.add_argument('--cores', type=int, choices=(1, 2), default=1)
    parser.add_argument('--spaces', type=lambda value: value.split(','), default=['vmem', 'cmem'])
    parser.add_argument('--iterations', type=lambda value: [int(item) for item in value.split(',')], default=[1, 3, 8, 17])
    parser.add_argument('--repeats', type=int, default=24)
    parser.add_argument('--seed', type=int, default=521)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/11_register_ring'))
    args = parser.parse_args()
    if min(args.streams, args.accumulators or args.streams) > 32:
        parser.error('超过 32 条展开读取时须明确指定最多 32 个累加器；旧版 U=64/128 对照使用 A=8 或 A=2')
    if args.size < 2 * args.streams * 4096 or (args.accumulators is not None and (args.accumulators < 1 or args.streams % min(args.accumulators, args.streams))):
        parser.error('工作集至少容纳两组读取，展开数须为有效累加器数的整数倍')
    if min(args.iterations) < 1 or max(args.iterations) * 8 * args.streams // min(args.streams, args.accumulators or args.streams) > 2**24:
        parser.error('正迭代次数且每个 float32 累加器的整数结果须精确')
    main(args)
