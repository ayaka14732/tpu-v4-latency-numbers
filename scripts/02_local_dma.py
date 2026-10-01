"""用 paired LCC 测量 TPU v4 六条本地 DMA 路径；保留 raw cycles 和完整 payload 核验。

使用本机一颗芯片的一个或两个 TC。CMEM 按 TC 分区，载体不使用 CMEM，无须修改 libtpu。
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from typing import cast

os.environ.setdefault('TPU_CHIPS_PER_PROCESS_BOUNDS', '2,2,1')
os.environ.setdefault('TPU_PROCESS_BOUNDS', '1,1,1')
os.environ.setdefault('TPU_VISIBLE_CHIPS', '0,1,2,3')

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxlib.xla_client import LoadedExecutable
import numpy as np
from tpuasm import BundleInsertion, assemble_listing, executable_programs, format_assembly, insert_executable_bundles, load_executable, parse_assembly, replace_executable_programs
from tpuasm.backends import select_backend

TARGET = 'tpu-v4-tc'
PATHS = ('hbm_vmem', 'vmem_hbm', 'hbm_cmem', 'cmem_hbm', 'cmem_vmem', 'vmem_cmem')

def bundle(instruction: str = '') -> str:
    return '{ ' + instruction + ' }\n'

def read(register: int) -> str:
    return bundle(f's0: srdreg.lcclo s{register} ; s1: srdreg.lcchi s{register + 5}')

def dma(destination: str, source: str, rows: int) -> str:
    return bundle(f's0: dma.simple {destination}, {source}, length={rows}, dst_flag=[sflag:52]')

def wait(rows: int, fence: bool = True) -> str:
    return bundle(('s0: sfence ; ' if fence else '') + f'misc: vwait.ge [sflag:52], {rows}')

def clear(rows: int) -> str:
    return bundle(f'misc: vsyncadd.s32 [sflag:52], {-rows}')

def copy(destination: str, source: str, rows: int) -> str:
    return dma(destination, source, rows) + wait(rows) + clear(rows) + bundle('s0: sfence')

def compile_carrier(rows: int, host: np.ndarray, device: jax.Device, cores: int = 1) -> tuple:
    shape = host.shape[-2:]

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(host.shape, jnp.uint32),
        mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=cores),
        scratch_types=(pltpu.VMEM(shape, jnp.uint32), pltpu.SemaphoreType.DMA),
        name=f'local_dma_lcc_carrier_{rows}',
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
    )
    def kernel(x_hbm: Ref, out_hbm: Ref, data: Ref, sem: Ref) -> None:
        core = jax.lax.axis_index('tc')
        input_ref = x_hbm if cores == 1 else x_hbm.at[core]
        output_ref = out_hbm if cores == 1 else out_hbm.at[core]
        pltpu.async_copy(input_ref, data, sem).wait()
        data[:8, :] = data[:8, :] ^ jnp.uint32(0x13579BDF)
        pltpu.async_copy(data, output_ref, sem).wait()

    x = jax.device_put(host, device)
    compiled = jax.jit(kernel, compiler_options={'xla_msa_enable': 'false', 'xla_tpu_vmem_scavenging_mode': 'NONE'}).lower(x).compile()
    expected = host.copy()
    expected[..., :8, :] ^= np.uint32(0x13579BDF)
    np.testing.assert_array_equal(np.asarray(compiled(x)), expected)
    raw = bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize())
    (record, index, image), = executable_programs(raw)
    source = format_assembly(image, target=TARGET)
    program = parse_assembly(source)
    (pc, marker), = [(pc, item) for pc, block in enumerate(program.bundles) for item in block.instructions if item.mnemonic == 'vxor.8x128.u32' and '0x13579bdf' in item.operands]
    assert len(program.bundles[pc].instructions) == 1
    nearby = [item for block in program.bundles[pc - 5:pc + 6] for item in block.instructions]
    # 此固定版本载体的 HBM 地址和 TC VMEM 分配；不将 s0/s1/s7 当成通用 ABI。
    transfers = [item for item in nearby if item.mnemonic == 'dma.simple']
    (input_dma, output_dma) = transfers
    assert input_dma.operands[0].startswith('[vmem:') and input_dma.operands[1].startswith('[hbm:'), input_dma
    assert output_dma.operands[0].startswith('[hbm:') and output_dma.operands[1] == input_dma.operands[0], output_dma
    assert all(item.operands[-1] == 'dst_flag=[sflag:52]' for item in transfers), transfers
    assert any(item.mnemonic == 'vld.8x128' and item.operands[1] == '[vmem:0x0]' for item in nearby)
    occupied = {int(value) for value in re.findall(r'\bs(\d+)\b', str(nearby))}
    available = [register for register in range(30, -1, -1) if register not in occupied]
    register_map = dict(zip(range(20, 31), available[:11], strict=True))
    original = f'{marker.slot}: {marker.mnemonic} ' + ', '.join(marker.operands)
    source = source.replace(original, f'{marker.slot}: vmov.8x128 {marker.operands[0]}, {marker.operands[2]}')
    raw = replace_executable_programs(raw, {(record, index): assemble_listing(source)})
    return compiled, x, raw, (record, index), pc, source, input_dma.operands[1], output_dma.operands[0], register_map

def fragment(
    path: str,
    rows: int,
    padded_rows: int,
    boundary: str,
    window: int = 1,
    input_hbm: str = '[hbm:s0]',
    output_hbm: str = '[hbm:s1]',
    register_map: dict[int, int] | None = None,
    cores: int = 1,
) -> str:
    source, destination = path.split('_')
    # 用虚拟标量编号区分载体传入地址与探针临时寄存器，最后一次性映射。
    input_register = int(input_hbm.split(':s')[1][:-1])
    output_register = int(output_hbm.split(':s')[1][:-1])
    input_hbm, output_hbm = '[hbm:s100]', '[hbm:s101]'
    # s23=CMEM 起点，s24=TC VMEM 源，s28=TC VMEM 目标；均以 512 B 为单位。
    total_rows = window * rows
    setup = bundle('s0: sfence') + bundle('s1: sld s23, [smem:0x1]')
    setup += bundle(f's0: smul.u32 s23, {padded_rows}, s23 ; s1: simm.s32 s24, 0')
    setup += bundle(f's0: simm.s32 s28, {padded_rows}')
    if source == 'cmem':
        setup += copy('[cmem:s23]', '[vmem:s24]', total_rows)
    elif destination == 'cmem':
        setup += copy('[cmem:s23]', '[vmem:s28]', total_rows)
    setup += bundle('s0: sfence')
    src = {'hbm': input_hbm, 'vmem': '[vmem:s24]', 'cmem': '[cmem:s23]'}[source]
    dst = {'hbm': output_hbm, 'vmem': '[vmem:s28]', 'cmem': '[cmem:s23]'}[destination]
    if destination == 'hbm':
        setup += copy(output_hbm, '[vmem:s28]', total_rows)
    if cores == 2:
        # 沿用载体入口/出口的同芯片 TC rendezvous 地址构造。sflag 45 在入口已经消费，
        # 在此各 signal/wait 一次并归还计数，避免一个 TC 仍在准备 CMEM 时另一个已开始计时。
        setup += bundle('s1: sld s20, [smem:0x0]') + bundle('s1: sld s21, [smem:0x1]')
        setup += bundle('s0: sand.u32 s20, 0xfff, s20 ; s1: ssub.s32 s21, 1, s21')
        setup += bundle('s0: sshll.u32 s20, s20, 0x12 ; s1: sshll.u32 s21, s21, 0xe')
        setup += bundle('s0: sor.u32 s20, s20, s21') + bundle('s0: sor.u32 s20, 0x802d, s20')
        setup += bundle('misc: vsyncadd.remote.s32 [sflag:s20], 1')
        setup += bundle('s0: sfence ; misc: vwait.ge [sflag:45], 1')
        setup += bundle('misc: vsyncadd.s32 [sflag:45], -1') + bundle('s0: sfence')
    if window > 1:
        setup += bundle(f's0: smov s29, {src.split(":")[1][:-1]} ; s1: smov s30, {dst.split(":")[1][:-1]}')
    work = read(20)
    for slot in range(window):
        work += dma(f'[{destination}:s30]', f'[{source}:s29]', rows) if window > 1 else dma(dst, src, rows)
        if slot < window - 1:
            work += bundle(f's0: sadd.s32 s29, {rows}, s29 ; s1: sadd.s32 s30, {rows}, s30')
    work += read(21) + wait(total_rows, fence=boundary == 'complete') + read(22)
    # 负对照也必须在回写/回读前完成 DMA；这里只把 fence 移到 END 后。
    work += bundle('s0: sfence') + clear(total_rows) + bundle('s0: sfence')
    if destination != 'vmem':
        work += copy('[vmem:s28]', dst, total_rows)
    # 所有计数器广播和记录均在三个端点之后，完整输出另由载体写回。
    for tile, register in enumerate((20, 21, 22, 25, 26, 27)):
        work += bundle(f'va0: vmov.8x128 v12, s{register}') + bundle('misc: vnop') * 8
        work += bundle(f'vst: vst.8x128 [vmem:0x{2 * padded_rows + 8 * tile:x}], v12')
    work += bundle('s0: sfence')
    mapping = {**(register_map or {register: register for register in range(20, 31)}), 100: input_register, 101: output_register}
    text = f'.target {TARGET}\n' + setup + work
    return re.sub(r'(?<![\w.])s(\d+)\b(?!:)', lambda match: f's{mapping[int(match[1])]}', text)

def main(args: argparse.Namespace) -> None:
    device = jax.local_devices()[0]
    assert device.device_kind == 'TPU v4'
    assert version('libtpu') == '0.0.49'
    args.output.mkdir(parents=True, exist_ok=True)
    backend, _ = select_backend(TARGET)
    environment = {'python': sys.version, 'jax': jax.__version__, 'jaxlib': version('jaxlib'), 'libtpu': version('libtpu'), 'libtpu_build_id': backend.build_id, 'tpuasm': version('tpuasm')}
    environment.update(device=str(device), device_kind=device.device_kind, repeats=args.repeats, seed=args.seed, boundary=args.boundary, cores=args.cores, window=args.window, protocol='local_dma_v2_poisoned_destination')
    environment['pre_measurement_rendezvous'] = args.cores == 2
    (args.output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    records = []
    rng = np.random.default_rng(args.seed)
    with (args.output / 'results.jsonl').open('w') as stream:
        for size in args.sizes:
            rows = size // 512
            total_rows = rows * args.window
            padded_rows = max(8, (total_rows + 7) // 8 * 8)
            shape = (2 * padded_rows + 48, 128) if args.cores == 1 else (args.cores, 2 * padded_rows + 48, 128)
            host = rng.integers(0, 1 << 32, shape, dtype=np.uint32)
            host[..., padded_rows:2 * padded_rows, :] = 0xDEADBEEF
            compiled, x, raw, key, pc, listing, input_hbm, output_hbm, register_map = compile_carrier(rows, host, device, args.cores)
            (args.output / f'carrier-{size}.tpuasm').write_text(listing)
            for path in args.paths:
                name = f'{path}-{size}-w{args.window}-c{args.cores}-{args.boundary}'
                probe = fragment(path, rows, padded_rows, args.boundary, args.window, input_hbm, output_hbm, register_map, args.cores)
                patched = insert_executable_bundles(raw, {key: [BundleInsertion(pc, probe)]})
                (_, _, image), = executable_programs(patched)
                full = format_assembly(image, target=TARGET)
                # format_assembly 自身会重汇编并逐字节核验。
                (args.output / f'{name}.tpuasm').write_text(probe)
                (args.output / f'{name}.full.tpuasm').write_text(full)
                function = load_executable(patched, compiled)
                counters = []
                for repeat in range(args.repeats + 2):
                    actual = np.asarray(function(x)).reshape(args.cores, 2 * padded_rows + 48, 128)
                    # 检查完整源、目标和未触及 padding；计数器的 48 行逐 word 检查一致。
                    source_host = host.reshape(args.cores, 2 * padded_rows + 48, 128)
                    expected = source_host[:, :2 * padded_rows].copy()
                    expected[:, padded_rows:padded_rows + total_rows] = source_host[:, :total_rows]
                    np.testing.assert_array_equal(actual[:, :2 * padded_rows], expected)
                    tiles = actual[:, 2 * padded_rows:].reshape(args.cores, 6, 8, 128)
                    np.testing.assert_array_equal(tiles, np.broadcast_to(tiles[:, :, :1, :1], tiles.shape))
                    halves = tiles[:, :, 0, 0].astype(np.uint64)
                    counts = halves[:, :3] | (halves[:, 3:] << np.uint64(32))
                    if repeat >= 2:
                        counters.append(counts.tolist())
                gaps = np.asarray(counters, dtype=np.uint64)[:, :, 1:] - np.asarray(counters, dtype=np.uint64)[:, :, :1]
                record = {'path': path, 'size_bytes': size, 'window': args.window, 'cores': args.cores, 'boundary': args.boundary, 'lcc': counters, 'gaps_from_begin': gaps.tolist(), 'correct': True}
                record.update(min_cycles=int(gaps[:, :, 1].min()), median_cycles=float(np.median(gaps[:, :, 1])), max_cycles=int(gaps[:, :, 1].max()))
                record['cycles_per_core'] = [{'min': int(values.min()), 'median': float(np.median(values)), 'max': int(values.max())} for values in gaps[:, :, 1].T]
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                records.append(record)
                print(name, 'issue', np.unique(gaps[:, :, 0]).tolist(), 'complete min/median/max', record['min_cycles'], record['median_cycles'], record['max_cycles'], flush=True)
            del function, compiled, x
            jax.clear_caches()
    (args.output / 'summary.json').write_text(json.dumps({'environment': environment, 'records': records}, indent=2) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=lambda value: [int(item) for item in value.split(',')], default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304])
    parser.add_argument('--paths', type=lambda value: value.split(','), default=list(PATHS))
    parser.add_argument('--repeats', type=int, default=24)
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--cores', type=int, choices=(1, 2), default=1)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--boundary', choices=('complete', 'issue_only'), default='complete')
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/02_local_dma'))
    args = parser.parse_args()
    if any(size < 512 or size % 512 or size > 4194304 for size in args.sizes) or any(path not in PATHS for path in args.paths) or args.repeats < 1:
        parser.error('大小须为 512 B 的整数倍且不超过 4 MiB；路径须来自 PATHS；repeats 须为正')
    if args.window < 1 or max(args.sizes) * args.window > 4194304:
        parser.error('窗口须为正且每 TC 的总 payload 不超过 4 MiB')
    main(args)
