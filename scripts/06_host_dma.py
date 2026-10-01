"""用发起 TC 的 paired LCC 测量 vint 请求到 pinned Host DMA 完成的周期。"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import re
from typing import cast

remote = import_module('04_remote_dma')  # 在导入 JAX 前设置本机四芯片范围。

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax.numpy as jnp
from jaxlib.xla_client import LoadedExecutable
import numpy as np
from tpuasm import BundleInsertion, executable_programs, format_assembly, insert_executable_bundles, load_executable, parse_assembly

def make_carrier(device: jax.Device, size: int, window: int) -> tuple:
    mesh = Mesh(np.asarray([device]), ('chip',))
    tc = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    device_sharding = NamedSharding(mesh, P(), memory_kind='device')
    host_sharding = NamedSharding(mesh, P(), memory_kind='pinned_host')

    @jax.jit(
        in_shardings=(device_sharding, device_sharding, host_sharding),
        out_shardings=(host_sharding, device_sharding),
        compiler_options=remote.OPTIONS,
    )
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P(), P()),
        out_specs=(P(), P()),
        check_vma=False,
    )
    def run(x: jax.Array, offset: jax.Array, poison: jax.Array) -> tuple[jax.Array, jax.Array]:
        source = jax.new_ref(x, memory_space=pltpu.HBM)
        target = jax.new_ref(poison, memory_space=pl.HOST)
        index = jax.new_ref(offset, memory_space=pltpu.HBM)
        record_hbm = jax.empty_ref(jax.ShapeDtypeStruct((128,), jnp.uint32), memory_space=pltpu.HBM)

        @pl.kernel(
            mesh=tc,
            scratch_types=(pltpu.SMEM((128,), jnp.uint32), pltpu.SMEM((128,), jnp.int32)),
            name=f'host_cycles_s{size}_w{window}',
            compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
        )
        def kernel(record: Ref, offset_smem: Ref) -> None:
            local = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
            sem = jax.empty_ref(jax.ShapeDtypeStruct((window,), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
            pltpu.async_copy(index, offset_smem, local).wait()
            for i in range(8):
                record[i] = jnp.uint32(0x12340000 + i)
            copies = [pltpu.make_async_copy(source.at[offset_smem[0] + w], target.at[w], sem.at[w]) for w in range(window)]
            for copy in copies:
                copy.start()
            for copy in copies:
                copy.wait()
            pltpu.async_copy(record, record_hbm, local).wait()

        kernel()
        return jax.freeze(target), jax.freeze(record_hbm)

    return run, device_sharding, host_sharding

def instrument(raw: bytes, output: Path, boundary: str, gap: int | None = None) -> tuple[bytes, dict]:
    (record, index, image), = executable_programs(raw)
    listing = format_assembly(image, target=remote.TARGET)
    (output / 'carrier.tpuasm').write_text(listing)
    program = parse_assembly(listing)
    operations = [(pc, item) for pc, block in enumerate(program.bundles) for item in block.instructions]
    marker = next(pc for pc, item in operations if item.mnemonic == 'simm.s32' and item.operands[1] == str(0x12340000))
    begin = next(pc for pc, item in operations if pc > marker and item.mnemonic == 'vint')
    end, dma = next((pc, item) for pc, item in operations if pc > begin and item.mnemonic == 'dma.simple' and item.operands[0].startswith('[hbm:') and item.operands[1].startswith('[smem:'))
    pointer = dma.operands[1][6:-1]
    _, definition = next((pc, item) for pc, item in reversed(operations) if begin < pc < end and item.mnemonic == 'simm.s32' and item.operands[0] == pointer)
    base = int(definition.operands[1], 0)
    if boundary == 'issue_only':
        end = max(pc for pc, item in operations if begin <= pc < end and item.mnemonic == 'vint') + 1
    region = [item for pc, item in operations if begin <= pc <= end]
    occupied = {int(value) for item in region for operand in item.operands for value in re.findall(r'\bs(\d+)\b', operand)}
    free = [value for value in range(30, -1, -1) if value not in occupied][:4]
    assert dma.operands[2] == 'length=1'
    b = remote.bundle
    storage = 'registers' if len(free) == 4 else 'smem'
    if storage == 'registers':
        prefix = b('s0: sfence') + ''.join(b(f's1: sst [smem:0x{base + 8 + i:x}], s{r}') for i, r in enumerate(free))
        prefix += b(f's0: srdreg.lcclo s{free[0]} ; s1: srdreg.lcchi s{free[1]}')
        suffix = b('s0: sfence') if boundary == 'complete' else ''
        suffix += b(f's0: srdreg.lcclo s{free[2]} ; s1: srdreg.lcchi s{free[3]}') + b('s0: sfence')
        suffix += ''.join(b(f's1: sst [smem:0x{base + i:x}], s{r}') for i, r in enumerate(free))
        suffix += ''.join(b(f's1: sld s{r}, [smem:0x{base + 8 + i:x}]') for i, r in enumerate(free)) + b() * 3
    else:
        # W=16 的载体没有四个空闲 SREG。只在端点临时借用两个寄存器，随后恢复。
        # raw 区间包含 BEGIN 的记录/恢复和 END 的现场保存，不能当作零开销探针。
        free = [30, 29]
        save = ''.join(b(f's1: sst [smem:0x{base + 8 + i:x}], s{r}') for i, r in enumerate(free))
        restore = ''.join(b(f's1: sld s{r}, [smem:0x{base + 8 + i:x}]') for i, r in enumerate(free)) + b() * 3
        read = b('s0: srdreg.lcclo s30 ; s1: srdreg.lcchi s29')
        prefix = b('s0: sfence') + save + read
        prefix += ''.join(b(f's1: sst [smem:0x{base + i:x}], s{r}') for i, r in enumerate(free)) + restore
        suffix = (b('s0: sfence') if boundary == 'complete' else '') + save + read + b('s0: sfence')
        suffix += ''.join(b(f's1: sst [smem:0x{base + 2 + i:x}], s{r}') for i, r in enumerate(free)) + restore
    if gap is None:
        edits = [BundleInsertion(begin, f'.target {remote.TARGET}\n' + prefix), BundleInsertion(end, f'.target {remote.TARGET}\n' + suffix)]
    else:
        edits = [BundleInsertion(end, f'.target {remote.TARGET}\n' + prefix + b() * gap + suffix)]
    patched = insert_executable_bundles(raw, {(record, index): edits})
    (_, _, image), = executable_programs(patched)
    (output / 'instrumented.tpuasm').write_text(format_assembly(image, target=remote.TARGET))
    audit = {'begin_pc': begin, 'end_pc': end, 'record_base_word': base, 'counter_sregs': free, 'boundary': boundary, 'counter_storage': storage, 'gap_bundles': gap}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return patched, audit

def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    run, device_sharding, host_sharding = make_carrier(jax.local_devices()[0], args.size, args.window)
    slots = 64 * 1024**2 // args.size
    host = np.random.default_rng(args.seed).integers(0, 1 << 32, (slots, args.size // 512, 128), dtype=np.uint32)
    poison = jax.device_put(np.full((args.window, args.size // 512, 128), remote.SENTINEL, np.uint32), host_sharding)
    x = jax.device_put(host, device_sharding)
    offset = jax.device_put(np.zeros(128, dtype=np.int32), device_sharding)
    compiled = run.lower(x, offset, poison).compile()
    payload, records = compiled(x, offset, poison)
    assert payload.sharding.memory_kind == 'pinned_host'
    np.testing.assert_array_equal(np.asarray(payload), host[:args.window])
    np.testing.assert_array_equal(np.asarray(records)[:8], np.arange(8, dtype=np.uint32) + 0x12340000)
    raw = bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize())
    (args.output / 'carrier.executable').write_bytes(raw)
    print('baseline correct', flush=True)
    patched, audit = instrument(raw, args.output, args.boundary, args.gap)
    function = load_executable(patched, compiled)
    rows = []
    for repeat in range(args.repeats + 2):
        start = (repeat * 17) % (slots - args.window + 1)
        index = np.zeros(128, dtype=np.int32)
        index[0] = start
        offset = jax.device_put(index, device_sharding)
        payload, record = function(x, offset, poison)
        assert payload.sharding.memory_kind == 'pinned_host'
        np.testing.assert_array_equal(np.asarray(payload), host[start:start + args.window])
        words = np.asarray(record)
        np.testing.assert_array_equal(words[4:8], np.arange(4, 8, dtype=np.uint32) + 0x12340000)
        lcc = [int(words[0]) | int(words[1]) << 32, int(words[2]) | int(words[3]) << 32]
        if repeat >= 2:
            rows.append({'source_slot': start, 'lcc': lcc, 'cycles': lcc[1] - lcc[0], 'full_payload_correct': True, 'pinned_host_verified': True})
    payload, _ = compiled(x, offset, poison)
    np.testing.assert_array_equal(np.asarray(payload), host[start:start + args.window])
    np.testing.assert_array_equal(np.asarray(poison), np.full(poison.shape, remote.SENTINEL, np.uint32))
    result = {
        'size_bytes': args.size,
        'window': args.window,
        'seed': args.seed,
        'working_set_bytes': host.nbytes,
        'audit': audit,
        'records': rows,
        'libtpu': remote.version('libtpu'),
        'jax': jax.__version__,
        'baseline_after': True,
        'poison_input_unchanged': True,
    }
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('cycles', [record['cycles'] for record in rows], flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=4096)
    parser.add_argument('--window', type=int, choices=(1, 4, 16), default=1)
    parser.add_argument('--repeats', type=int, default=24)
    parser.add_argument('--seed', type=int, default=461)
    parser.add_argument('--boundary', choices=('complete', 'issue_only'), default='complete')
    parser.add_argument('--gap', type=int)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/06_host_dma'))
    args = parser.parse_args()
    if args.size < 4096 or args.size % 4096 or args.size * args.window > 64 * 1024**2:
        parser.error('size 须为 4 KiB 整倍数且单窗口至多 64 MiB')
    main(args)
