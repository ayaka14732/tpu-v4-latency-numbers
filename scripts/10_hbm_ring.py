"""在每 TC 的 64 MiB HBM 环形工作集中测四条本地路径，直接记录 paired LCC。"""

from __future__ import annotations

import argparse
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import re
from typing import cast

local = import_module('02_local_dma')
jax, jnp, np, pl, pltpu = local.jax, local.jnp, local.np, local.pl, local.pltpu
Ref = local.Ref
Mesh, NamedSharding, P = jax.sharding.Mesh, jax.sharding.NamedSharding, jax.sharding.PartitionSpec
RING_BYTES = 64 * 1024**2
POISON = 0xDEADBEEF
PATHS = ('hbm_vmem', 'vmem_hbm', 'hbm_cmem', 'cmem_hbm')

def carrier(total_rows: int, cores: int, device: jax.Device) -> object:
    shape = (2 * total_rows + 48, 128)
    tc = pltpu.TensorCoreMesh(axis_name='tc', num_cores=cores)
    mesh = Mesh(np.array([device]), ('chip',))
    sharding = NamedSharding(mesh, P(), memory_kind='device')

    @jax.jit(
        donate_argnums=(2,),
        in_shardings=sharding,
        out_shardings=(sharding, sharding),
        compiler_options={'xla_msa_enable': 'false', 'xla_tpu_vmem_scavenging_mode': 'NONE'},
    )
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P(), P()),
        out_specs=(P(), P()),
        check_vma=False,
    )
    def run(x: jax.Array, offsets: jax.Array, poison: jax.Array) -> tuple[jax.Array, jax.Array]:
        source = jax.new_ref(x, memory_space=pltpu.HBM)
        target = jax.new_ref(poison)
        index = jax.new_ref(offsets, memory_space=pltpu.HBM)
        records = jax.empty_ref(jax.ShapeDtypeStruct((cores, *shape), jnp.uint32), memory_space=pltpu.HBM)

        @pl.kernel(
            mesh=tc,
            scratch_types=(pltpu.VMEM(shape, jnp.uint32), pltpu.SMEM((128,), jnp.int32), pltpu.SemaphoreType.DMA),
            name=f'hbm_ring_c{cores}_r{total_rows}',
            compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
        )
        def kernel(data: Ref, offset_smem: Ref, sem: Ref) -> None:
            core = jax.lax.axis_index('tc')
            pltpu.async_copy(index.at[core], offset_smem, sem).wait()
            start = offset_smem[0]
            pltpu.async_copy(source.at[core, pl.ds(start, total_rows)], data.at[:total_rows], sem).wait()
            pltpu.async_copy(target.at[core, pl.ds(start, total_rows)], data.at[total_rows:2 * total_rows], sem).wait()
            data[2 * total_rows:] = jnp.zeros((48, 128), dtype=jnp.uint32)
            data[:8] = data[:8] ^ jnp.uint32(0x13579BDF)
            pltpu.async_copy(data.at[:total_rows], target.at[core, pl.ds(start, total_rows)], sem).wait()
            pltpu.async_copy(data, records.at[core], sem).wait()

        kernel()
        return jax.freeze(target), jax.freeze(records)

    return run

def prepare(raw: bytes, output: Path, total_rows: int) -> tuple[bytes, tuple[int, int], int, str, str, dict[int, int], int, dict]:
    (record, index, image), = local.executable_programs(raw)
    listing = local.format_assembly(image, target=local.TARGET)
    (output / 'carrier.tpuasm').write_text(listing)
    program = local.parse_assembly(listing)
    operations = [(pc, item) for pc, block in enumerate(program.bundles) for item in block.instructions]
    (marker_pc, marker), = [(pc, item) for pc, item in operations if item.mnemonic == 'vxor.8x128.u32' and '0x13579bdf' in item.operands]

    def constant_before(pointer: str, before: int) -> int:
        register = pointer.split(':')[1][:-1]
        if not register.startswith('s'):
            return int(register, 0)
        _, instruction = next((pc, item) for pc, item in reversed(operations) if pc < before and item.operands and item.operands[0] == register)
        assert instruction.mnemonic == 'simm.s32', instruction
        return int(instruction.operands[1], 0)

    transfers = [(pc, item) for pc, item in operations if item.mnemonic == 'dma.simple']
    index_pc, index_dma = next((pc, item) for pc, item in reversed(transfers) if pc < marker_pc and item.operands[0].startswith('[smem:') and item.operands[1].startswith('[hbm:'))
    base = constant_before(index_dma.operands[0], index_pc)
    (input_pc, input_dma), = [(pc, item) for pc, item in transfers if index_pc < pc < marker_pc and item.operands[0].startswith('[vmem:') and constant_before(item.operands[0], pc) == 0]
    output_pc, output_dma = next((pc, item) for pc, item in transfers if pc > marker_pc and item.operands[0].startswith('[hbm:') and item.operands[1].startswith('[vmem:'))
    assert constant_before(output_dma.operands[1], output_pc) == 0
    for item in (input_dma, output_dma):
        assert item.operands[2] == f'length={total_rows}' and item.operands[3] == 'dst_flag=[sflag:52]', item
    input_register = int(input_dma.operands[1].split(':s')[1][:-1])
    output_register = int(output_dma.operands[0].split(':s')[1][:-1])
    borrowed = [register for register in range(30, -1, -1) if register != output_register][:12]
    mapping = dict(zip(range(20, 31), borrowed[:11], strict=True))
    capture = f'.target {local.TARGET}\n' + local.bundle(f's1: sst [smem:0x{base + 79:x}], s{input_register}')
    original = f'{marker.slot}: {marker.mnemonic} ' + ', '.join(marker.operands)
    listing = listing.replace(original, f'{marker.slot}: vmov.8x128 {marker.operands[0]}, {marker.operands[2]}')
    raw = local.replace_executable_programs(raw, {(record, index): local.assemble_listing(listing)})
    audit = dict(input_pc=input_pc, output_pc=output_pc, index_smem_base=base, input_pointer_register=input_register, output_pointer_register=output_register, borrowed_registers=borrowed)
    return raw, (record, index), output_pc, capture, output_dma.operands[0], mapping, base, audit

def check(payload: np.ndarray, record: np.ndarray, source: np.ndarray, starts: list[int], total_rows: int, baseline: bool) -> list[list[int]]:
    counts = []
    for core, start in enumerate(starts):
        expected = source[core, start:start + total_rows].copy()
        if baseline:
            expected[:8] ^= np.uint32(0x13579BDF)
        np.testing.assert_array_equal(payload[core, :start], np.uint32(POISON))
        np.testing.assert_array_equal(payload[core, start:start + total_rows], expected)
        np.testing.assert_array_equal(payload[core, start + total_rows:], np.uint32(POISON))
        np.testing.assert_array_equal(record[core, :total_rows], expected)
        np.testing.assert_array_equal(record[core, total_rows:2 * total_rows], np.uint32(POISON) if baseline else expected)
        if baseline:
            np.testing.assert_array_equal(record[core, 2 * total_rows:], 0)
        else:
            tiles = record[core, 2 * total_rows:].reshape(6, 8, 128)
            np.testing.assert_array_equal(tiles, np.broadcast_to(tiles[:, :1, :1], tiles.shape))
            halves = tiles[:, 0, 0].astype(np.uint64)
            counts.append((halves[:3] | (halves[3:] << np.uint64(32))).tolist())
    return counts

def main(args: argparse.Namespace) -> None:
    assert version('libtpu') == '0.0.49'
    device = jax.local_devices()[0]
    assert device.device_kind == 'TPU v4'
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    source = rng.integers(0, 1 << 32, (args.cores, RING_BYTES // 512, 128), dtype=np.uint32)
    x = jax.device_put(source, device)
    poison_host = np.full_like(source, POISON)

    def fresh_poison() -> jax.Array:
        return jax.device_put(poison_host, device)

    all_records = []
    for size in args.sizes:
        total_rows = size // 512 * args.window
        slots = RING_BYTES // (total_rows * 512)
        starts = [0] * args.cores
        offsets = np.zeros((args.cores, 128), dtype=np.int32)
        index = jax.device_put(offsets, device)
        run = carrier(total_rows, args.cores, device)
        poison = fresh_poison()
        compiled = run.lower(x, index, poison).compile()
        payload, record = compiled(x, index, poison)
        check(np.asarray(payload), np.asarray(record), source, starts, total_rows, baseline=True)
        print('baseline', size, 'correct', flush=True)
        output = args.output / f's{size}'
        output.mkdir(exist_ok=True)
        raw = bytes(cast(local.LoadedExecutable, compiled.runtime_executable()).serialize())
        (output / 'carrier.executable').write_bytes(raw)
        if args.carrier_only:
            (_, _, image), = local.executable_programs(raw)
            (output / 'carrier.tpuasm').write_text(local.format_assembly(image, target=local.TARGET))
            continue
        raw, key, pc, capture, output_hbm, mapping, base, audit = prepare(raw, output, total_rows)
        borrowed = audit['borrowed_registers']
        for path in args.paths:
            save = ''.join(local.bundle(f's1: sst [smem:0x{base + 80 + i:x}], s{register}') for i, register in enumerate(borrowed))
            restore = ''.join(local.bundle(f's1: sld s{register}, [smem:0x{base + 80 + i:x}]') for i, register in enumerate(borrowed)) + local.bundle() * 3
            load = local.bundle(f's1: sld s{borrowed[-1]}, [smem:0x{base + 79:x}]') + local.bundle() * 3
            probe = local.fragment(path, size // 512, total_rows, 'complete', args.window, f'[hbm:s{borrowed[-1]}]', output_hbm, mapping, args.cores)
            probe = probe.replace(f'.target {local.TARGET}\n', f'.target {local.TARGET}\n' + local.bundle('s0: sfence') + save + load, 1) + restore
            edits = [local.BundleInsertion(audit['input_pc'], capture), local.BundleInsertion(pc, probe)]
            patched = local.insert_executable_bundles(raw, {key: edits})
            (_, _, image), = local.executable_programs(patched)
            (output / f'{path}.full.tpuasm').write_text(local.format_assembly(image, target=local.TARGET))
            (output / f'{path}.tpuasm').write_text(probe)
            function = local.load_executable(patched, compiled)
            samples = []
            for repeat in range(args.repeats + 2):
                stride = max(1, slots // 17) | 1
                first = repeat * stride % slots if repeat < args.repeats - 1 else (repeat - args.repeats) % slots
                starts = [first * total_rows] * args.cores
                offsets[:, 0] = starts
                index = jax.device_put(offsets, device)
                payload, record = function(x, index, fresh_poison())
                counts = check(np.asarray(payload), np.asarray(record), source, starts, total_rows, baseline=False)
                if repeat >= 2:
                    samples.append(dict(source_slot=first, lcc=counts, cycles=[row[2] - row[0] for row in counts], full_payload_and_guards_correct=True))
            row = dict(path=path, size_bytes=size, window=args.window, cores=args.cores, working_set_bytes_per_core=RING_BYTES, ring_slots=slots, records=samples)
            all_records.append(row)
            print(path, size, 'median', np.median([sample['cycles'] for sample in samples], axis=0).tolist(), flush=True)
        payload, record = compiled(x, index, fresh_poison())
        check(np.asarray(payload), np.asarray(record), source, starts, total_rows, baseline=True)
        np.testing.assert_array_equal(poison_host, np.uint32(POISON))
        np.testing.assert_array_equal(np.asarray(x), source)
        (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        del compiled, run, function
        jax.clear_caches()
    summary = dict(protocol='local_hbm_ring_lcc_v1', libtpu=version('libtpu'), seed=args.seed, baseline_after=not args.carrier_only, fresh_poison_donated_each_call=True, source_unchanged=True, records=all_records)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=lambda value: [int(item) for item in value.split(',')], default=[4096])
    parser.add_argument('--paths', type=lambda value: value.split(','), default=list(PATHS))
    parser.add_argument('--cores', type=int, choices=(1, 2), default=1)
    parser.add_argument('--window', type=int, choices=(1, 4, 8), default=1)
    parser.add_argument('--repeats', type=int, default=24)
    parser.add_argument('--seed', type=int, default=511)
    parser.add_argument('--carrier-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/10_hbm_ring'))
    args = parser.parse_args()
    if any(size < 4096 or size % 4096 or size * args.window > 4194304 for size in args.sizes) or any(path not in PATHS for path in args.paths) or args.repeats < 1:
        parser.error('大小须为 4 KiB 的整数倍，每窗口至多 4 MiB，路径须为四条 HBM 本地路径且 repeats 为正')
    main(args)
