"""用 BC 本地 paired LCC 测量 TC VMEM→私有 BMEM，保留 TC 源的 ready/release 协议。"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
from importlib import import_module
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import cast

bc = import_module('05_bc_dma')

from jax_allgather_marker import configure_jax
from jax_pallas_barnacore_wakeup_ab_probe import configure_final_bundle_dump
from jax_pallas_tc_sflag_window_probe import install_observer as install_tc, read_barrier_resolution, remove_observer as remove_tc
from pallas_vmem_capability import install_llo_allocation_observer, read_llo_allocation_placements, remove_llo_allocation_observer

import numpy as np

COMPILER_OPTIONS = {'xla_msa_enable': 'false', 'xla_tpu_vmem_scavenging_mode': 'NONE', 'xla_mosaic_unsafe_allow_multicore_remote_dma': 'true'}

def make_source(device: object, space: str, size: int) -> tuple[Callable[[object], object], object]:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    mesh = Mesh(np.array([device]), ('chip',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='core', num_cores=1)
    sharding = NamedSharding(mesh, P(), memory_kind='device')

    @jax.jit(
        in_shardings=sharding,
        out_shardings=sharding,
        compiler_options=COMPILER_OPTIONS,
    )
    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def run(x: object) -> object:
        source = jax.new_ref(x, memory_space=pltpu.HBM)
        scratch = (pltpu.CMEM((size // 512, 128), x.dtype),) if space == 'cmem' else ()

        @pl.kernel(
            mesh=tc_mesh,
            name='bc_pull_source',
            scratch_types=scratch,
            compiler_params=pltpu.CompilerParams(
                collective_id=37,
                has_side_effects=pltpu.SideEffectType.SIDE_EFFECTING,
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(*cmem: object) -> None:
            tile = jax.empty_ref(jax.ShapeDtypeStruct((size // 512, 128), x.dtype), memory_space=pltpu.VMEM @ tc_mesh)
            sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
            barrier = pltpu.get_barrier_semaphore()
            pltpu.async_copy(source, tile, sem).wait()
            if cmem:
                pltpu.async_copy(tile, cmem[0], sem).wait()
            # 发布前的 TC 读取保持 staging 顺序；BC 的 source-completion credit 保持本次调用中的分配存活。
            pl.semaphore_signal(barrier, tile[0, 0] * 0 + 1, core_index=2)
            pl.semaphore_wait(barrier, size // 512)

        kernel()
        return jax.freeze(source)

    return run, sharding

def program_source(size: int, window: int) -> str:
    text = bc.window_program('vmem', size, window).replace('registers=5-6,10-15,20-', 'registers=4-6,10-19,20-')
    text = text.replace('%s10 = sld [barna_core_smem:7200]', '%s4 = sld [barna_core_smem:7212]\n%s10 = sld [barna_core_smem:7200]')
    begin = text.index('hot_loop:\n') + len('hot_loop:\n')
    end = text.index('.scalar_block advance_count')
    total = 15 if window == 1 else 28
    poison = '.scalar_block poison_flag\n%s13 = smov 9\nsst [%s13] %s12\n.endscalar_block\n'
    poison += bc.bundle('.s1 sdone.write %s13, %s12') + bc.bundle() * 3
    poison += bc.bundle(f'.s0 scalar_dma_simple c1:v=15 c2:v=1 c4:v=0 a1:v=4 a2:v={total} a3:v=12 a4:v=13 a5:v=0 a6:v=0 a7:v=0 a8:v=0 a9:v=0 a10:v=1 a11:v=0 a12:v=0 a13:v=0 a14:v=0 a15:v=0 a16:v=0 a17:v=0')
    poison += bc.bundle('.s0 swait.done %s13') + bc.bundle('.s0 scalar_fence c1:v=15 c2:v=1 c4:v=0 a1:v=32 a2:v=255 a3:v=0 a4:v=0 a5:v=0')
    poison += '.scalar_block restore_payload_flag\n%s13 = smov 10\n.endscalar_block\n'
    after = bc.paired(18, 19) + bc.bundle() * 4
    after += '.scalar_block publish_counters\n' + '\n'.join(f'sst [{7208 + i}] %s{16 + i}' for i in range(4)) + '\n.endscalar_block\n'
    return (text[:begin] + poison + bc.paired(16, 17) + text[begin:end] + after + text[end:]).replace('.pad_to 128', '.pad_to 256')

def main(args: argparse.Namespace) -> None:
    assert bc.version('libtpu') == '0.0.46'
    for name in bc.RESIDENT_REPLACEMENT_ENVIRONMENT:
        os.environ.pop(name, None)
    args.output.mkdir(parents=True, exist_ok=True)
    size, window = args.size, args.window
    total = size * window
    source_path = args.output / 'program.bca'
    source_path.write_text(program_source(size, window))
    spec = bc.load_spec(bc.default_spec_path())
    bindings = {'SOURCE_CORE': 2, 'SOURCE_MEMORY': 0} if args.space == 'vmem' else {'SOURCE_CORE': 1, 'SOURCE_MEMORY': 2}
    image = bc.encode_semantic_object(bc.parse_source(source_path, spec), spec, bindings)
    (args.output / 'program.pb').write_bytes(image)
    configure_jax()
    dumps = configure_final_bundle_dump()
    shim_path = args.output.parent / 'bc_pull_lcc_shim.so'
    if not shim_path.exists():
        bc.build_shim(shim_path)
    global _LIBTPU_OWNER
    _LIBTPU_OWNER = bc.ctypes.CDLL(str(bc.current_libtpu_path()))
    shim = bc.ctypes.CDLL(str(shim_path))
    adapter = bc.LibtpuProcessAdapter(shim, bc.SUPPORTED_TARGET)
    owner = 'bc-pull-paired-lcc-v1'

    def allocation(name: str, address: int, space: bc.MemorySpace = bc.MemorySpace.BARNA_CORE_SMEM) -> bc.BarnaCoreAllocation:
        return bc.BarnaCoreAllocation(name, space, address, bc.AddressUnit.WORD32, 4, 4, owner)

    wake = allocation('wake', 1, bc.MemorySpace.BARNA_CORE_SFLAG)
    program = bc.BarnaCoreProgram.from_semantic_proto(image, mailbox_abi=owner, required_allocations=(wake,), shutdown_request=bc.BarnaCoreShutdownRequest('wake', (2).to_bytes(4, 'little')))
    registration = adapter.register_program(program)
    code, message = bc.install_observer(shim, bc.current_libtpu_path(), args.output / 'observer.txt')
    if code:
        raise RuntimeError(message)
    import jax
    from jax._src import tpu_custom_call
    from jaxlib.xla_client import LoadedExecutable

    device = jax.local_devices()[0]
    wrappers = adapter.devices()
    target = next(d for d in wrappers if d.core_id == 0 and tuple(d.chip_coordinates[:2]) == tuple(device.coords[:2]))
    fields = {name: allocation(name, 7200 + i) for i, name in enumerate(('source', 'output', 'count', 'end', 'size', 'step', 'tc_flag', 'release_source'))}
    fields.update({f'counter{i}': allocation(f'counter{i}', 7208 + i) for i in range(4)})
    fields['poison'] = allocation('poison', 7212)
    fields.update({name: allocation(name, index, bc.MemorySpace.BARNA_CORE_SFLAG) for name, index in (('wake', 1), ('result', 2), ('completion', 10), ('return_completion', 11))})

    def write(name: str, value: int) -> None:
        adapter.write(target, fields[name], int(value).to_bytes(4, 'little'))

    def read(name: str) -> int:
        return int.from_bytes(adapter.read(target, fields[name], size_bytes=4), 'little')

    deadline = time.monotonic() + 20
    for wrapper in wrappers:
        while int.from_bytes(adapter.read(wrapper, allocation('boot', 7230), size_bytes=4), 'little') != 1:
            if time.monotonic() > deadline:
                raise TimeoutError('BC bootstrap did not publish readiness')
    rng = np.random.default_rng(args.seed)
    run, sharding = make_source(device, 'vmem', total)
    host = rng.integers(0, 1 << 30, (total // 512, 128), dtype=np.int32)
    x = jax.device_put(host, sharding)
    x.block_until_ready()
    install_tc(shim, bc.current_libtpu_path())
    install_llo_allocation_observer(shim, bc.current_libtpu_path())
    # 当前 JAX 默认生成 Mosaic IR v17；固定 0.0.46 明确只接受到 v15。
    # 使用 JAX 自带的 serde 降级通道，不仅仅改写版本标签。
    previous_ir_override = tpu_custom_call.ir_version_override
    tpu_custom_call.ir_version_override = lambda: 15
    try:
        compiled = run.lower(x).compile()
        barrier, flag, calls = read_barrier_resolution(shim)
        placements = read_llo_allocation_placements(shim)
    finally:
        tpu_custom_call.ir_version_override = previous_ir_override
        remove_llo_allocation_observer(shim)
        remove_tc(shim)
    (args.output / 'barrier_resolution.json').write_text(json.dumps({'barrier': barrier, 'flag': flag, 'calls': calls, 'placements': [asdict(p) for p in placements]}, indent=2) + '\n')
    assert calls and barrier == 0 and flag not in ({0, 1, 2, 9, 10, 11} | (set(range(16, 16 + window)) if window > 1 else set())), (barrier, flag, calls)
    files = [path for path in dumps.rglob('*-final_bundles.txt') if 'bc_pull_source' in path.name and 'schedule-analysis' not in path.name]
    assert len(files) == 1, files
    listing = files[0].read_text()
    (args.output / 'source_final_bundles.txt').write_text(listing)
    raw = bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize())
    (args.output / 'source.executable').write_bytes(raw)
    (operand,) = set(re.findall(r'dma\.\w+_to_vmem[^\n]*?/\*vmem=\*/(%s\w+)', listing))
    (ordinal,) = set(re.findall(rf'{re.escape(operand)} = smov \[#allocation(\d+)\]', listing))
    (source,) = [p for p in placements if p.ordinal == int(ordinal) and p.size_bytes == total and p.memory_space == 3 and not p.dematerialized]
    assert source.offset_bytes >= 0 and source.offset_bytes % 512 == 0
    assert 'dma.vmem_to_hbm' not in listing
    (args.output / 'source_allocation.json').write_text(json.dumps({'source': asdict(source), 'tc_barrier': flag}, indent=2) + '\n')
    if args.space == 'cmem':
        # 运行时保持 0.0.46；只在独立 CPU 进程中使用当前 0.0.49 codec 编辑 TC 指令。
        # 原 TC VMEM 分配仍存活，最终 source release 也仍从该分配发出。
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(bc.ROOT.parent / 'tpuasm' / 'src')
        environment.pop('TPU_LIBRARY_PATH', None)
        output = args.output / 'cmem_source.executable'
        command = [sys.executable, str(Path(__file__).with_name('07_bc_source.py')), '--source', str(args.output / 'source.executable'), '--output', str(output)]
        command += ['--source-sreg', operand.rsplit('_s', 1)[1], '--source-row', str(source.offset_bytes // 512), '--size', str(total)]
        subprocess.run(command, env=environment, check=True)
        sys.path.insert(0, str(bc.ROOT.parent / 'tpuasm' / 'src'))
        from tpuasm import load_executable
        compiled = load_executable(output.read_bytes(), compiled)
    handles = adapter.adopt_bootstrap_program(program, registration)
    for handle in handles:
        adapter.shutdown_program(handle)
    deadline = time.monotonic() + 20
    while any(state.running for state in adapter.run_states()):
        if time.monotonic() > deadline:
            raise TimeoutError('BC initial stop did not complete')
    hbm = adapter.allocate(target, bc.MemorySpace.HBM, name='poison-output', size_bytes=2 * total + 1024, alignment_bytes=512)
    guard = bytes([0xD3]) * 512
    poison = np.full(total // 4, 0xDEADBEEF, np.uint32).tobytes()
    request = dict(source=source.offset_bytes // 512, output=hbm.byte_address // 512 + total // 512 + 1, count=1, end=source.offset_bytes // 512 + total // 512, size=size // 512)
    request.update(step=0, tc_flag=flag, release_source=source.offset_bytes // 512, poison=hbm.byte_address // 512)
    for name, value in request.items():
        write(name, value)
    handle = next(handle for handle in handles if handle.device == target)
    adapter.start_program(handle)
    rows = []
    try:
        for repeat in range(args.repeats + 2):
            host = rng.integers(0, 1 << 30, host.shape, dtype=np.int32)
            x = jax.device_put(host, sharding)
            x.block_until_ready()
            adapter.write(target, hbm, poison + guard + poison + guard)
            write('result', 0)
            result = compiled(x)
            write('wake', 1)
            deadline = time.monotonic() + 20
            while read('result') != 1:
                if time.monotonic() > deadline:
                    raise TimeoutError(f'BC request incomplete: {adapter.run_states()}')
            result.block_until_ready()
            np.testing.assert_array_equal(np.asarray(result), host)
            actual = adapter.read(target, hbm, offset_bytes=total, size_bytes=total + 1024)
            assert actual == guard + host.tobytes() + guard
            assert read('completion') == total // 512 and read('return_completion') == total // 512
            halves = [read(f'counter{i}') for i in range(4)]
            lcc = [halves[0] | halves[1] << 32, halves[2] | halves[3] << 32]
            if repeat >= 2:
                rows.append({'lcc': lcc, 'cycles': lcc[1] - lcc[0], 'full_payload_and_guards_correct': True, 'tc_source_released': True})
    finally:
        adapter.shutdown_program(handle)
        assert adapter.wait(handle, 15).state is bc.EventState.COMPLETE
        adapter.free(target, hbm)
        code, message = bc.remove_observer(shim)
        if code:
            raise RuntimeError(message)
        adapter.clear_program(registration)
    data = dict(protocol=owner, libtpu=bc.version('libtpu'), jax=jax.__version__, size_bytes=size, window=window, space=args.space, seed=args.seed, source=asdict(source), tc_flag=flag, records=rows, residents_stopped=True, hbm_freed=True, observer_removed=True)
    (args.output / 'summary.json').write_text(json.dumps(data, indent=2) + '\n')
    print('cycles', [row['cycles'] for row in rows], flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=4096)
    parser.add_argument('--window', type=int, choices=(1, 4, 16), default=1)
    parser.add_argument('--space', choices=('vmem', 'cmem'), default='vmem')
    parser.add_argument('--repeats', type=int, default=24)
    parser.add_argument('--seed', type=int, default=481)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/07_bc_pull'))
    args = parser.parse_args()
    if args.size < 4096 or args.size % 4096 or args.size * args.window > 2097152:
        parser.error('每条为 4 KiB 整倍数，窗口至多 2 MiB')
    main(args)
