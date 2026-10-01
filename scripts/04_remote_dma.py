"""编译带独立 SMEM 记录区的远程 DMA 载体，并插入同 TC paired LCC 完成端点。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
from typing import cast

os.environ.setdefault('TPU_CHIPS_PER_PROCESS_BOUNDS', '2,2,1')
os.environ.setdefault('TPU_PROCESS_BOUNDS', '1,1,1')
os.environ.setdefault('TPU_VISIBLE_CHIPS', '0,1,2,3')

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax.numpy as jnp
from jaxlib.xla_client import LoadedExecutable
import numpy as np
from tpuasm import BundleInsertion, executable_programs, format_assembly, insert_executable_bundles, load_executable, parse_assembly, replace_executable_programs
from tpuasm.tc_solver import BundleSolver
from tpuasm.tpu_v4_tc_assembler import ISA, _decoded_source
from tpuasm.tpu_v4_tc_codec import encode_program

TARGET = 'tpu-v4-tc'
SENTINEL = 0xDEADBEEF
OPTIONS = {'xla_msa_enable': 'false', 'xla_tpu_vmem_scavenging_mode': 'NONE', 'xla_mosaic_unsafe_allow_multicore_remote_dma': 'true'}

def bundle(instruction: str = '') -> str:
    return '{ ' + instruction + ' }\n'

def resize_payload(raw: bytes, capacity: int, size: int) -> tuple[bytes, list[dict]]:
    """保留载体布局，只替换 payload DMA 长度、对应 wait 和 completion 清零数。"""
    (record, index, image), = executable_programs(raw)
    program, encoded = _decoded_source(image)
    remote = [pc for pc, block in enumerate(program.bundles) for item in block.instructions if item.mnemonic == 'dma.general' and all(o.startswith(('[vmem:', '[cmem:')) for o in item.operands[:2])]
    end = next(pc for pc, block in enumerate(program.bundles) if pc > remote[-1] and any(i.mnemonic == 'dma.simple' for i in block.instructions))
    output = [(word, tuple(forms)) for word, forms in encoded]
    edits = []
    # 最后一次 completion 清零可能与输出 DMA 同 bundle；只改该 flag 操作，保留完整输出长度。
    for pc in range(remote[0], end + 1):
        block = program.bundles[pc]
        items = []
        for item in block.instructions:
            operands = item.operands
            if item.mnemonic == 'dma.general':
                assert pc in remote and operands[2] == f'length={capacity // 512}'
                operands = operands[:2] + (f'length={size // 512}',) + operands[3:]
            elif item.mnemonic in ('vwait.ge', 'vsyncadd.s32'):
                expected = capacity // 512 * (1 if item.mnemonic == 'vwait.ge' else -1)
                if operands[1] == str(expected):
                    operands = (operands[0], str(size // 512 * (1 if expected > 0 else -1)))
            if operands != item.operands:
                edits.append({'pc': pc, 'mnemonic': item.mnemonic, 'before': item.operands, 'after': operands})
                item = replace(item, operands=operands)
            items.append(item)
        if tuple(items) != block.instructions:
            output[pc] = BundleSolver(ISA, replace(block, instructions=tuple(items)), program, pc).solve()
    if size != capacity:
        assert len(edits) == 5 * len(remote), (size, capacity, len(edits), len(remote))
    return replace_executable_programs(raw, {(record, index): encode_program(output)}), edits

def instrument(raw: bytes, key: tuple[int, int], listing: str, output: Path, boundary: str, space: str = 'vmem', capacity: int = 4096, window: int = 1, chips: int = 2, gap: int | None = None) -> tuple[bytes, dict]:
    program = parse_assembly(listing)
    operations = [(pc, item) for pc, block in enumerate(program.bundles) for item in block.instructions]
    # 只匹配 kernel 的 TC VMEM→TC VMEM remote payload，避开 runtime 自身的 HBM DMA。
    remote = [(pc, item) for pc, item in operations if item.mnemonic == 'dma.general' and all(operand.startswith('[vmem:') for operand in item.operands[:2])]
    assert remote, 'missing payload remote DMA'
    begin = remote[0][0]
    end, output_dma = next((pc, item) for pc, item in operations if pc > remote[-1][0] and item.mnemonic == 'dma.simple' and item.operands[0].startswith('[hbm:') and item.operands[1].startswith('[vmem:'))
    record_pc, record_dma = next((pc, item) for pc, item in operations if pc > end and item.mnemonic == 'dma.simple' and item.operands[0].startswith('[hbm:') and item.operands[1].startswith('[smem:'))
    pointer = record_dma.operands[1].removeprefix('[smem:').removesuffix(']')
    _, definition = next((pc, item) for pc, item in reversed(operations) if end <= pc < record_pc and item.mnemonic == 'simm.s32' and item.operands[0] == pointer)
    base = int(definition.operands[1], 0)
    assert record_dma.operands[2] == 'length=1' and base % 128 == 64, record_dma
    region = [item for pc, item in operations if begin <= pc < end]
    occupied = {int(value) for item in region for operand in item.operands for value in re.findall(r'\bs(\d+)\b', operand)}
    free = [value for value in range(30, -1, -1) if value not in occupied][:4]
    storage = 'registers' if len(free) == 4 else 'smem'
    if storage == 'smem':
        free = [30, 29]
    # 四寄存器模式全程保留计数；大窗口的 SMEM 模式在端点借用并恢复活跃寄存器。
    prefix = bundle('s0: sfence')
    for offset, register in enumerate(free):
        prefix += bundle(f's1: sst [smem:0x{base + 8 + offset:x}], s{register}')
    if storage == 'registers':
        prefix += bundle(f's0: srdreg.lcclo s{free[0]} ; s1: srdreg.lcchi s{free[1]}')
        suffix = bundle('s0: sfence') if boundary == 'complete' else ''
        suffix += bundle(f's0: srdreg.lcclo s{free[2]} ; s1: srdreg.lcchi s{free[3]}')
        suffix += bundle('s0: sfence')
        for offset, register in enumerate(free):
            suffix += bundle(f's1: sst [smem:0x{base + offset:x}], s{register}')
        for offset, register in enumerate(free):
            suffix += bundle(f's1: sld s{register}, [smem:0x{base + 8 + offset:x}]')
        suffix += bundle() * 3
    else:
        # 大窗口占满 SREG 时，只在两个端点借用两个寄存器。与第 6 节相同，raw 区间保留 spill 成本。
        save = ''.join(bundle(f's1: sst [smem:0x{base + 8 + i:x}], s{r}') for i, r in enumerate(free))
        restore = ''.join(bundle(f's1: sld s{r}, [smem:0x{base + 8 + i:x}]') for i, r in enumerate(free)) + bundle() * 3
        read = bundle('s0: srdreg.lcclo s30 ; s1: srdreg.lcchi s29')
        prefix += read + ''.join(bundle(f's1: sst [smem:0x{base + i:x}], s{r}') for i, r in enumerate(free)) + restore
        suffix = (bundle('s0: sfence') if boundary == 'complete' else '') + save + read + bundle('s0: sfence')
        suffix += ''.join(bundle(f's1: sst [smem:0x{base + 2 + i:x}], s{r}') for i, r in enumerate(free)) + restore
    edits = [BundleInsertion(begin, f'.target {TARGET}\n' + prefix), BundleInsertion(end, f'.target {TARGET}\n' + suffix)]
    ici_overrides = []
    if space in ('cmem', 'cmem_setup'):
        # 在原 ready rendezvous 之前初始化两个 CMEM 区域，防止接收端 poison 覆盖已经到达的 payload。
        ready_signals = [pc for pc, item in operations if pc < begin and item.mnemonic == 'vsyncadd.remote.s32'][-chips:]
        assert len(ready_signals) == chips
        flag = output_dma.operands[3].removeprefix('dst_flag=')
        rows = capacity * window // 512

        def copy(destination: str, source: str) -> str:
            body = bundle(f's0: dma.simple {destination}, {source}, length={rows}, dst_flag={flag}')
            return body + bundle(f'misc: vwait.ge {flag}, {rows}') + bundle(f'misc: vsyncadd.s32 {flag}, -{rows}') + bundle('s0: sfence')

        temp = free[0]
        setup = bundle('s0: sfence') + bundle(f's1: sst [smem:0x{base + 12:x}], s{temp}')
        addresses = []
        for operand in remote[0][1].operands[:2]:
            pointer = operand[6:-1]
            _, definition = next((pc, item) for pc, item in reversed(operations) if pc < begin and item.operands and item.operands[0] == pointer)
            assert definition.mnemonic == 'simm.s32' and definition.predicate == 15, definition
            address = int(definition.operands[1], 0)
            addresses.append(address)
            setup += bundle(f's0: simm.s32 s{temp}, {address}') + copy(f'[cmem:s{temp}]', f'[vmem:s{temp}]')
        assert abs(addresses[0] - addresses[1]) >= rows
        setup += bundle(f's1: sld s{temp}, [smem:0x{base + 12:x}]') + bundle() * 3
        edits.insert(0, BundleInsertion(ready_signals[0], f'.target {TARGET}\n' + setup))
        if space == 'cmem':
            suffix += copy(output_dma.operands[1], output_dma.operands[1].replace('[vmem:', '[cmem:'))
            edits[-1] = BundleInsertion(end, f'.target {TARGET}\n' + suffix)
            (_, _, image), = executable_programs(raw)
            decoded, encoded = _decoded_source(image)
            words = [(word, tuple(forms)) for word, forms in encoded]
            # libtpu 0.0.49 DmaGeneralOverrides 对 CMEM 不写 TC destination override。
            # VMEM TC0 的 bits[28:26]=2 必须清零，否则远端仍被定向到 TC。
            override_edits = {}
            for pc, dma in remote:
                register = next(o.removeprefix('ici_dest=') for o in dma.operands if o.startswith('ici_dest='))
                definition_pc, definition = next((p, item) for p, item in reversed(operations) if p < pc and item.operands and item.operands[0] == register)
                assert definition.mnemonic == 'simm.s32' and definition.predicate == 15, definition
                before = int(definition.operands[1], 0) & 0xffffffff
                assert (before >> 26) & 7 == 2, hex(before)
                after = before & ~(7 << 26)
                override_edits[definition_pc] = (register, after if after < 2**31 else after - 2**32)
                ici_overrides.append({'dma_pc': pc, 'definition_pc': definition_pc, 'register': register, 'before': hex(before), 'after': hex(after)})
            for pc, (register, immediate) in override_edits.items():
                block = decoded.bundles[pc]
                items = tuple(replace(item, operands=(register, str(immediate))) if item.mnemonic == 'simm.s32' and item.operands[0] == register else item for item in block.instructions)
                words[pc] = BundleSolver(ISA, replace(block, instructions=items), decoded, pc).solve()
            for pc, _ in remote:
                block = decoded.bundles[pc]
                items = tuple(replace(item, operands=tuple(o.replace('[vmem:', '[cmem:') for o in item.operands)) if item.mnemonic == 'dma.general' else item for item in block.instructions)
                words[pc] = BundleSolver(ISA, replace(block, instructions=items), decoded, pc).solve()
            raw = replace_executable_programs(raw, {key: encode_program(words)})
    if gap is not None:
        # 保留真实 DMA/credit 与 CMEM 准备，仅把计数器放到完成之后校准空区间。
        edits = [edit for edit in edits if edit.image_pc not in (begin, end)]
        edits.append(BundleInsertion(end, f'.target {TARGET}\n' + prefix + bundle() * gap + suffix))
    patched = insert_executable_bundles(raw, {key: edits})
    (_, _, image), = executable_programs(patched)
    (output / 'instrumented.tpuasm').write_text(format_assembly(image, target=TARGET))
    audit = {'begin_pc': begin, 'end_pc': end, 'record_base_word': base, 'counter_sregs': free, 'counter_storage': storage, 'gap_bundles': gap, 'remote_dma_count': len(remote), 'boundary': boundary, 'space': space, 'ici_overrides': ici_overrides}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return patched, audit

def make_carrier(devices: list[jax.Device], size: int, window: int, flows: list[tuple[int, int]], cores: int, round_trip: bool = False) -> tuple:
    mesh = Mesh(np.asarray(devices), ('chip',))
    tc = pltpu.TensorCoreMesh(axis_name='tc', num_cores=cores)
    shape = (window, size // 512, 128)
    record_shape = (128,)

    @pl.kernel(
        out_type=(jax.ShapeDtypeStruct((1, cores, *shape), jnp.uint32), jax.ShapeDtypeStruct((1, cores, *record_shape), jnp.uint32)),
        mesh=tc,
        scratch_types=(pltpu.SMEM(record_shape, jnp.uint32),),
        name=f'remote_cycles_s{size}_w{window}',
        compiler_params=pltpu.CompilerParams(collective_id=37, disable_bounds_checks=True, disable_semaphore_checks=True),
    )
    def kernel(x_hbm: Ref, out_hbm: Ref, record_hbm: Ref, record: Ref) -> None:
        chip = jax.lax.axis_index('chip')
        core = jax.lax.axis_index('tc')
        node = chip * cores + core
        send = jax.empty_ref(jax.ShapeDtypeStruct(shape, jnp.uint32), memory_space=pltpu.VMEM @ tc)
        recv = jax.empty_ref(jax.ShapeDtypeStruct(shape, jnp.uint32), memory_space=pltpu.VMEM @ tc)
        local_sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
        send_sem = jax.empty_ref(jax.ShapeDtypeStruct((window,), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
        recv_sem = jax.empty_ref(jax.ShapeDtypeStruct((window,), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
        credit = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.REGULAR.dtype), memory_space=pltpu.SEMAPHORE)
        pltpu.async_copy(x_hbm.at[0, core], send, local_sem).wait()
        recv[...] = jnp.full(shape, jnp.uint32(SENTINEL), jnp.uint32)
        for index in range(8):
            record[index] = jnp.uint32(0x12340000 + index)
        ready = pltpu.get_barrier_semaphore()
        for target in range(len(devices) * cores):
            pl.semaphore_signal(ready, 1, device_id={'chip': target // cores, 'tc': target % cores})
        pl.semaphore_wait(ready, len(devices) * cores)
        for slot in range(window):
            for source, destination in flows:
                @pl.when(node == source)
                def start() -> None:
                    pltpu.make_async_remote_copy(send.at[slot], recv.at[slot], send_sem.at[slot], recv_sem.at[slot], device_id={'chip': destination // cores, 'tc': destination % cores}).start()
        for slot in range(window):
            for source, destination in flows:
                transfer = pltpu.make_async_remote_copy(send.at[slot], recv.at[slot], send_sem.at[slot], recv_sem.at[slot], device_id={'chip': destination // cores, 'tc': destination % cores})
                @pl.when(node == source)
                def wait_send() -> None:
                    transfer.wait_send()
                @pl.when(node == destination)
                def wait_recv() -> None:
                    transfer.wait_recv()
        if round_trip:
            reply_send = jax.empty_ref(jax.ShapeDtypeStruct((window,), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
            reply_recv = jax.empty_ref(jax.ShapeDtypeStruct((window,), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
            for source, destination in flows:
                for slot in range(window):
                    @pl.when(node == destination)
                    def reply() -> None:
                        pltpu.make_async_remote_copy(recv.at[slot], recv.at[slot], reply_send.at[slot], reply_recv.at[slot], device_id={'chip': source // cores, 'tc': source % cores}).start()
                for slot in range(window):
                    transfer = pltpu.make_async_remote_copy(recv.at[slot], recv.at[slot], reply_send.at[slot], reply_recv.at[slot], device_id={'chip': source // cores, 'tc': source % cores})
                    @pl.when(node == destination)
                    def wait_reply_send() -> None:
                        transfer.wait_send()
                    @pl.when(node == source)
                    def wait_reply_recv() -> None:
                        transfer.wait_recv()
        else:
            for source, destination in flows:
                @pl.when(node == destination)
                def release() -> None:
                    pl.semaphore_signal(credit, 1, device_id={'chip': source // cores, 'tc': source % cores})
            for source, _ in flows:
                @pl.when(node == source)
                def acquire() -> None:
                    pl.semaphore_wait(credit, 1)
        pltpu.async_copy(recv, out_hbm.at[0, core], local_sem).wait()
        pltpu.async_copy(record, record_hbm.at[0, core], local_sem).wait()

    @jax.jit(compiler_options=OPTIONS)
    @jax.shard_map(
        mesh=mesh,
        in_specs=P('chip'),
        out_specs=(P('chip'), P('chip')),
        check_vma=False,
    )
    def run(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return kernel(x)

    return run, NamedSharding(mesh, P('chip'))

def main(args: argparse.Namespace) -> None:
    assert version('libtpu') == '0.0.49'
    devices = list(jax.local_devices())[:args.chips]
    assert len(devices) == args.chips and all(device.device_kind == 'TPU v4' for device in devices)
    flows = [tuple(map(int, flow.split(':'))) for flow in args.flows.split(',')]
    assert all(0 <= source < len(devices) * args.cores and 0 <= destination < len(devices) * args.cores and source != destination for source, destination in flows)
    assert len({source for source, _ in flows}) == len(flows) and len({destination for _, destination in flows}) == len(flows)
    if args.round_trip:
        assert len({node for flow in flows for node in flow}) == 2 * len(flows), 'RTT 并发 pair 的节点须互不重叠'
    args.output.mkdir(parents=True, exist_ok=True)
    host = np.random.default_rng(args.seed).integers(0, 1 << 32, (len(devices), args.cores, args.window, args.size // 512, 128), dtype=np.uint32)
    run, sharding = make_carrier(devices, args.size, args.window, flows, args.cores, args.round_trip)
    x = jax.device_put(host, sharding)
    compiled = run.lower(x).compile()
    expected = np.full_like(host, SENTINEL)
    for source, destination in flows:
        expected[destination // args.cores, destination % args.cores] = host[source // args.cores, source % args.cores]
        if args.round_trip:
            expected[source // args.cores, source % args.cores] = host[source // args.cores, source % args.cores]
    payload, records = compiled(x)
    np.testing.assert_array_equal(np.asarray(payload), expected)
    np.testing.assert_array_equal(np.asarray(records).reshape(-1, 128)[:, :8], np.broadcast_to(np.arange(8, dtype=np.uint32) + 0x12340000, (len(devices) * args.cores, 8)))
    raw = bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize())
    (args.output / 'carrier.executable').write_bytes(raw)
    (record, index, image), = executable_programs(raw)
    listing = format_assembly(image, target=TARGET)
    (args.output / 'carrier.tpuasm').write_text(listing)
    program = parse_assembly(listing)
    operations = [{'pc': pc, 'slot': item.slot, 'mnemonic': item.mnemonic, 'operands': item.operands, 'predicate': item.predicate} for pc, block in enumerate(program.bundles) for item in block.instructions]
    (args.output / 'operations.json').write_text(json.dumps(operations, indent=2) + '\n')
    (args.output / 'baseline.json').write_text(json.dumps({'correct': True, 'flows': flows, 'size_bytes': args.size, 'window': args.window, 'devices': [str(device) for device in devices], 'cores': args.cores}, indent=2) + '\n')
    print('baseline correct; carrier and operations saved', flush=True)
    patched, audit = instrument(raw, (record, index), listing, args.output, args.boundary, args.space, args.size, args.window, args.chips, args.gap)
    results = []
    for size in args.sizes or [args.size]:
        resized, edits = resize_payload(patched, args.size, size)
        function = load_executable(resized, compiled)
        partial = expected.copy()
        partial[..., size // 512:, :] = SENTINEL
        counters = []
        for repeat in range(args.repeats + 2):
            payload, records = function(x)
            np.testing.assert_array_equal(np.asarray(payload), partial)
            words = np.asarray(records).reshape(-1, 128)
            np.testing.assert_array_equal(words[:, 4:8], np.broadcast_to(np.arange(4, 8, dtype=np.uint32) + 0x12340000, (len(devices) * args.cores, 4)))
            halves = words[:, :4].astype(np.uint64)
            values = halves[:, (0, 2)] | (halves[:, (1, 3)] << np.uint64(32))
            if repeat >= 2:
                counters.append(values.tolist())
        cycles = np.asarray(counters, dtype=np.uint64)[:, :, 1] - np.asarray(counters, dtype=np.uint64)[:, :, 0]
        result = {
            'size_bytes': size,
            'capacity_bytes': args.size,
            'window': args.window,
            'space': args.space,
            'round_trip': args.round_trip,
            'flows': flows,
            'devices': [str(device) for device in devices],
            'cores': args.cores,
            'audit': audit,
            'length_edits': edits,
            'lcc': counters,
            'cycles': cycles.tolist(),
            'correct': True,
        }
        result['source_distributions'] = {str(source): {'min': int(cycles[:, source].min()), 'median': float(np.median(cycles[:, source])), 'max': int(cycles[:, source].max())} for source, _ in flows}
        (args.output / f'result_{size}.json').write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
        print(size, json.dumps(result['source_distributions']), flush=True)
    payload, _ = compiled(x)
    np.testing.assert_array_equal(np.asarray(payload), expected)
    protocol = 'remote_lcc_v2_payload_rtt' if args.round_trip else 'remote_lcc_v3_all_credits_first'
    summary = {'protocol': protocol, 'libtpu': version('libtpu'), 'jax': jax.__version__, 'seed': args.seed, 'repeats': args.repeats, 'records': results, 'baseline_after': True}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=4096)
    parser.add_argument('--sizes', type=lambda value: [int(item) for item in value.split(',')])
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--chips', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--cores', type=int, choices=(1, 2), default=1)
    parser.add_argument('--flows', default='0:1')
    parser.add_argument('--seed', type=int, default=421)
    parser.add_argument('--repeats', type=int, default=24)
    parser.add_argument('--boundary', choices=('complete', 'issue_only'), default='complete')
    parser.add_argument('--gap', type=int, help='真实传输结束后测量 M 个空 bundle，仅用于端点校准')
    parser.add_argument('--space', choices=('vmem', 'cmem', 'cmem_setup'), default='vmem')
    parser.add_argument('--round-trip', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/04_remote_dma'))
    args = parser.parse_args()
    if args.round_trip and args.space != 'vmem':
        parser.error('payload RTT 当前仅支持 TC VMEM')
    if args.space != 'vmem' and args.cores != 1:
        parser.error('Megacore Shared CMEM 远程探针每芯片使用 TC0')
    if args.sizes:
        args.size = max(args.size, max(args.sizes))
        if any(size < 4096 or size % 4096 for size in args.sizes):
            parser.error('sizes 须为至少 4 KiB 的整倍数')
    if args.size < 4096 or args.size % 4096 or args.window < 1 or args.size * args.window > 4194304:
        parser.error('每条大小须为 4 KiB 整数倍，每 TC payload 不超过 4 MiB')
    main(args)
