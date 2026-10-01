"""在固定 libtpu 0.0.46 process-zero runtime 上测量 HBM→BC 私有 BMEM 的本地 LCC 周期。"""

from __future__ import annotations

import argparse
import ctypes
from collections.abc import Callable
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TPU_CHIPS_PER_PROCESS_BOUNDS', '2,2,1')
os.environ.setdefault('TPU_PROCESS_BOUNDS', '1,1,1')
os.environ.setdefault('TPU_VISIBLE_CHIPS', '0,1,2,3')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT.parent / 'tpu-v4-barnacore-support' / 'barnacore_assembler'
SUPPORT = ROOT / 'scripts' / 'support'
sys.path[:0] = [str(BACKEND), str(SUPPORT)]

from assembler import default_spec_path, encode_semantic_object, load_spec, parse_source
from backend import AddressUnit, BarnaCoreAllocation, BarnaCoreProgram, BarnaCoreShutdownRequest, EventState, MemorySpace
from bc_window import window_program
from jax_allgather_marker import build_shim, current_libtpu_path, install_observer, remove_observer
from libtpu_backend import LibtpuProcessAdapter, SUPPORTED_TARGET
from process_zero_runtime_probe import current_process_accel_fds, RESIDENT_REPLACEMENT_ENVIRONMENT

def bundle(instructions: str = '') -> str:
    return '.bundle\n' + instructions + ('\n' if instructions else '') + '.endbundle\n'

def paired(low: int, high: int) -> str:
    return bundle(f'.s0 scalar_read_registers c1:v=15 c2:v=1 c4:v=0 a1:v=0 a2:v={low}\n.s1 scalar_read_registers c1:v=15 c2:v=1 c4:v=0 a1:v=1 a2:v={high}')

def source(size: int, window: int, gap: int | None) -> str:
    text = window_program('hbm', size, window).replace('registers=5-6,10-15,20-', 'registers=4-6,10-19,20-')
    text = text.replace('%s10 = sld [barna_core_smem:7200]', '%s4 = sld [barna_core_smem:7206]\n%s10 = sld [barna_core_smem:7200]')
    hot = text.index('hot_loop:\n') + len('hot_loop:\n')
    # 每次请求先用独立 HBM poison 填满私有 BMEM；这段不计时。
    total_register = 15 if window == 1 else 28
    poison = '.scalar_block poison_flag\n%s13 = smov 8\nsst [%s13] %s12\n.endscalar_block\n'
    poison += bundle('.s1 sdone.write %s13, %s12') + bundle() * 3
    poison += bundle(f'.s0 scalar_dma_simple c1:v=15 c2:v=1 c4:v=0 a1:v=4 a2:v={total_register} a3:v=12 a4:v=13 a5:v=0 a6:v=0 a7:v=0 a8:v=0 a9:v=0 a10:v=1 a11:v=0 a12:v=0 a13:v=0 a14:v=0 a15:v=0 a16:v=0 a17:v=0')
    poison += bundle('.s0 swait.done %s13') + bundle('.s0 scalar_fence c1:v=15 c2:v=1 c4:v=0 a1:v=32 a2:v=255 a3:v=0 a4:v=0 a5:v=0')
    text = text[:hot] + poison + text[hot:]
    end = text.index('.scalar_block advance_count')
    begin = text.index('.bundle\n.s0 scalar_dma_simple', hot + len(poison))
    if window > 1:
        begin = text.index('.scalar_block address_window_0', hot + len(poison))
    if gap is not None:
        begin = end
    after = paired(18, 19) + bundle() * 4
    after += '.scalar_block publish_counters\n' + '\n'.join(f'sst [{7208 + i}] %s{16 + i}' for i in range(4)) + '\n.endscalar_block\n'
    interval = paired(16, 17) + (bundle() * gap if gap is not None else text[begin:end]) + after
    return (text[:begin] + interval + text[end:]).replace('.pad_to 128', '.pad_to 256')

def main(args: argparse.Namespace) -> None:
    assert 'jax' not in sys.modules and version('libtpu') == SUPPORTED_TARGET.libtpu_version == '0.0.46'
    for name in RESIDENT_REPLACEMENT_ENVIRONMENT:
        os.environ.pop(name, None)
    args.output.mkdir(parents=True, exist_ok=True)
    size, window = args.size, args.window
    total = size * window
    owner = 'bc-paired-lcc-v1'

    def allocation(name: str, space: MemorySpace, address: int) -> BarnaCoreAllocation:
        return BarnaCoreAllocation(name, space, address, AddressUnit.WORD32, 4, 4, owner)

    fields = {name: allocation(name, MemorySpace.BARNA_CORE_SMEM, 7200 + i) for i, name in enumerate(('source', 'output', 'count', 'end', 'size', 'step', 'poison'))}
    fields.update({f'counter{i}': allocation(f'counter{i}', MemorySpace.BARNA_CORE_SMEM, 7208 + i) for i in range(4)})
    fields.update({name: allocation(name, MemorySpace.BARNA_CORE_SFLAG, index) for name, index in (('wake', 1), ('completion', 8), ('return_completion', 9), ('result', 2))})
    source_path = args.output / 'program.bca'
    source_path.write_text(source(size, window, args.gap))
    spec = load_spec(default_spec_path())
    image = encode_semantic_object(parse_source(source_path, spec), spec, {})
    program = BarnaCoreProgram.from_semantic_proto(image, mailbox_abi=owner, required_allocations=tuple(fields.values()), shutdown_request=BarnaCoreShutdownRequest('wake', (2).to_bytes(4, 'little')))
    (args.output / 'program.pb').write_bytes(image)
    shim_path = args.output.parent / 'bc_lcc_shim.so'
    if not shim_path.exists():
        build_shim(shim_path)
    global _LIBTPU_OWNER
    _LIBTPU_OWNER = ctypes.CDLL(str(current_libtpu_path()))
    shim = ctypes.CDLL(str(shim_path))
    adapter = LibtpuProcessAdapter(shim, SUPPORTED_TARGET)
    registration = adapter.register_program(program)
    installed, initialized = False, False
    handle, hbm, devices = None, None, ()
    errors, primary = [], None
    rows = []
    try:
        code, message = install_observer(shim, current_libtpu_path(), args.output / 'observer.txt')
        if code:
            raise RuntimeError(message)
        installed = True
        adapter.initialize_process_zero(current_libtpu_path())
        initialized = True
        devices = adapter.devices()
        device = devices[0]
        measured_device = str(device)
        registration, _ = adapter.stage_current_program_shell(program, registration)
        handle = adapter.load_program(device, program)
        ring_size = (64 * 1024**2 - 2 * total - 1024) // total * total
        hbm = adapter.allocate(device, MemorySpace.HBM, name='ring-poison-output', size_bytes=ring_size + 2 * total + 1024, alignment_bytes=512)
        base = hbm.byte_address // 512
        rng = np.random.default_rng(args.seed)
        data = rng.integers(0, 256, ring_size, dtype=np.uint8).tobytes()
        poison = np.full(total // 4, 0xDEADBEEF, dtype=np.uint32).tobytes()
        guard = bytes([0xD3]) * 512
        adapter.write(device, hbm, data + poison + guard + poison + guard)

        def write(name: str, value: int) -> None:
            adapter.write(device, fields[name], int(value).to_bytes(4, 'little'))

        def read(name: str) -> int:
            return int.from_bytes(adapter.read(device, fields[name], size_bytes=4), 'little')

        request = dict(
            source=base,
            output=base + (ring_size + total + 512) // 512,
            count=1,
            end=base + ring_size // 512,
            size=size // 512,
            step=size // 512,
            poison=base + ring_size // 512,
            wake=0,
            result=0,
        )
        for name, value in request.items():
            write(name, value)
        adapter.start_program(handle)
        for repeat in range(args.repeats + 2):
            # 跨不同地址取一次完整窗口，源 payload 和地址均变化，回环也由取模显式定义。
            offset = ((repeat * 17) % (ring_size // total)) * total
            if repeat >= args.repeats - 1:
                offset = ((repeat - args.repeats) % (ring_size // total)) * total
            write('source', base + offset // 512)
            write('result', 0)
            adapter.write(device, hbm, guard + poison + guard, offset_bytes=ring_size + total)
            write('wake', 1)
            deadline = time.monotonic() + 15
            while read('result') != 1:
                if time.monotonic() > deadline:
                    raise TimeoutError(f'BC request did not complete: {adapter.run_states()}')
            halves = [read(f'counter{i}') for i in range(4)]
            lcc = [halves[0] | halves[1] << 32, halves[2] | halves[3] << 32]
            actual = adapter.read(device, hbm, offset_bytes=ring_size + total, size_bytes=total + 1024)
            assert actual == guard + data[offset:offset + total] + guard, 'BC payload or guard mismatch'
            assert read('return_completion') == total // 512
            if repeat >= 2:
                rows.append({'source_offset': offset, 'lcc': lcc, 'cycles': lcc[1] - lcc[0], 'full_payload_and_guards_correct': True})
        print('cycles', [row['cycles'] for row in rows], flush=True)
    except BaseException as error:
        primary = error
    finally:
        def cleanup(label: str, function: Callable[[], object]) -> None:
            try:
                function()
            except BaseException as error:
                errors.append(f'{label}: {error}')

        if handle is not None:
            def stop() -> None:
                if adapter.program_completion(handle).state is EventState.PENDING:
                    adapter.shutdown_program(handle)
                    assert adapter.wait(handle, 15).state is EventState.COMPLETE
            cleanup('stop', stop)
        if hbm is not None:
            cleanup('free HBM', lambda: adapter.free(handle.device, hbm))
        for device in devices:
            cleanup('destroy wrapper', lambda: adapter.destroy_independent_wrapper(device))
        if installed:
            def remove() -> None:
                code, message = remove_observer(shim)
                if code:
                    raise RuntimeError(message)
            cleanup('remove observer', remove)
        cleanup('clear program', lambda: adapter.clear_program(registration))
        if initialized:
            cleanup('shutdown runtime', adapter.shutdown_process_zero)
    if primary:
        primary.add_note('; '.join(errors))
        raise primary
    assert not errors and not current_process_accel_fds(), errors
    result = {
        'protocol': owner,
        'libtpu': version('libtpu'),
        'size_bytes': size,
        'window': window,
        'gap_bundles': args.gap,
        'seed': args.seed,
        'working_set_bytes': ring_size,
        'device': measured_device,
        'records': rows,
        'cleanup_clean': True,
    }
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=4096)
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--gap', type=int)
    parser.add_argument('--repeats', type=int, default=24)
    parser.add_argument('--seed', type=int, default=451)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/05_bc_dma'))
    args = parser.parse_args()
    if args.size < 512 or args.size % 512 or args.window not in (1, 4, 16) or args.size * args.window > 2097152:
        parser.error('512 B 整倍数，W=1/4/16，每窗口不超过 2 MiB')
    main(args)
