"""用同一 BC 的 paired LCC 包围 Host Magic Queue 窗口，保留并量化 Host/BC 端点握手。"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys

bc = import_module('05_bc_dma')
ctypes = bc.ctypes
np = bc.np

def counter_source() -> str:
    text = '.target pufferfish-bcs\n'
    text += bc.bundle('.s0 %s5 = smov sy.sel_hex_80000000') + bc.bundle('.s0 %s10 = smov sy.sel_zero') + bc.bundle('.s1 sst [28] %s10')

    def request() -> str:
        body = bc.bundle('.s0 swait.gt %s5, sy.sel_zero')
        body += bc.bundle('.s1 scalar_load_smem c1:v=15 c2:v=4 c4:v=0 a1:v=49 a2:v=6 a3:v=0 a4:v=0 a5:v=0 a6:v=0')
        return body + bc.bundle() * 3 + bc.bundle('.s0 %p0 = scmp.eq.s32.totalorder %s6, 2') + bc.bundle('.s0 sbr.rel (%p0) @exit') + bc.bundle()

    def acknowledge() -> str:
        # 先清 wake 再发布 sequence，下一次请求不会被旧一轮的清零覆盖。
        body = bc.bundle('.s0 %s8 = smov sy.sel_zero') + bc.bundle('.s1 sst [sy.sel_hex_80000000] %s8')
        return body + bc.bundle('.s0 %s10 = sadd.s32 %s10, 1') + bc.bundle('.s1 sst [28] %s10')

    text += 'begin:\n' + request() + bc.paired(16, 17) + acknowledge()
    text += request() + bc.paired(18, 19) + bc.bundle() * 4
    for i in range(4):
        text += bc.bundle(f'.s1 sst [{24 + i}] %s{16 + i}')
    text += acknowledge() + bc.bundle('.s0 sbr.rel @begin') + bc.bundle()
    return text + 'exit:\n' + bc.bundle('.s0 shalt') + '.pad_to 64\n'

def main(args: argparse.Namespace) -> None:
    assert 'jax' not in sys.modules and bc.version('libtpu') == '0.0.46'
    for name in bc.RESIDENT_REPLACEMENT_ENVIRONMENT:
        os.environ.pop(name, None)
    args.output.mkdir(parents=True, exist_ok=True)
    size, window = args.size, args.window
    total = size * window
    shim_path = args.output.parent / 'magic_lcc_shim.so'
    backend_source = bc.BACKEND.parent / 'tpu_embedding_oracle_tf218_libtpu218_20260820/jax_bridge/tpu_embedding_control_plane_shim.cc'
    base_helper = bc.SUPPORT / 'magic_queue_host.cc'
    helper = Path(__file__).with_suffix('.cc')
    combined = args.output.parent / 'magic_lcc_combined.cc'
    combined.write_text(''.join(f'#include "{path.resolve()}"\n' for path in (backend_source, base_helper, helper)))
    if not shim_path.exists() or shim_path.stat().st_mtime < max(path.stat().st_mtime for path in (backend_source, base_helper, helper)):
        subprocess.run(['g++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared', str(combined), '-ldl', '-o', str(shim_path)], check=True)
    source = args.output / 'counter.bca'
    source.write_text(counter_source())
    spec = bc.load_spec(bc.default_spec_path())
    image = bc.encode_semantic_object(bc.parse_source(source, spec), spec, {})
    (args.output / 'counter.pb').write_bytes(image)
    owner = 'host-magic-bc-lcc-envelope-v2'
    wake = bc.BarnaCoreAllocation('wake', bc.MemorySpace.BARNA_CORE_SFLAG, 1, bc.AddressUnit.WORD32, 4, 4, owner)
    program = bc.BarnaCoreProgram.from_semantic_proto(image, mailbox_abi=owner, required_allocations=(wake,), shutdown_request=bc.BarnaCoreShutdownRequest('wake', (2).to_bytes(4, 'little')))
    global _LIBTPU_OWNER
    _LIBTPU_OWNER = ctypes.CDLL(str(bc.current_libtpu_path()))
    shim = ctypes.CDLL(str(shim_path))
    shim.MagicHostCreate.argtypes = [ctypes.c_size_t, ctypes.c_uint64, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    shim.MagicHostWindowLcc.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
    shim.MagicHostClose.argtypes = [ctypes.c_void_p]
    adapter = bc.LibtpuProcessAdapter(shim, bc.SUPPORTED_TARGET)
    registration = adapter.register_program(program)
    installed, initialized = False, False
    devices, handle, hbm = (), None, None
    session, host_pointer = ctypes.c_void_p(), ctypes.c_void_p()
    errors, primary = [], None
    rows = []
    try:
        code, message = bc.install_observer(shim, bc.current_libtpu_path(), args.output / 'observer.txt')
        if code:
            raise RuntimeError(message)
        installed = True
        adapter.initialize_process_zero(bc.current_libtpu_path())
        initialized = True
        devices = adapter.devices()
        device = devices[0]
        measured_device = str(device)
        registration, _ = adapter.stage_current_program_shell(program, registration)
        handle = adapter.load_program(device, program)
        slots = 64 * 1024**2 // size
        hbm = adapter.allocate(device, bc.MemorySpace.HBM, name='magic-source-ring', size_bytes=slots * size, alignment_bytes=512)
        token = adapter._hbm_allocations[hbm.owner].allocation_token
        code = shim.MagicHostCreate(device.wrapper_id, token, total, ctypes.byref(session), ctypes.byref(host_pointer))
        if code:
            raise RuntimeError(f'MagicHostCreate: {code}')
        source_bytes = np.random.default_rng(args.seed).integers(0, 256, slots * size, dtype=np.uint8).tobytes()
        adapter.write(device, hbm, source_bytes)
        adapter.write(device, wake, bytes(4))
        adapter.start_program(handle)
        sequence = ctypes.c_uint32()
        guard = bytes([0xD3]) * 4096
        for repeat in range(args.repeats + 2):
            first = repeat * 17 % slots
            if repeat >= args.repeats - 1:
                first = (repeat - args.repeats) % slots
            ctypes.memset(host_pointer.value, 0xD3, total + 8192)
            endpoints = (ctypes.c_uint64 * 2)()
            code = shim.MagicHostWindowLcc(session, first, slots, window, ctypes.byref(sequence), endpoints, args.empty)
            if code:
                raise RuntimeError(f'MagicHostWindowLcc: {code}')
            actual = ctypes.string_at(host_pointer, total + 8192)
            indices = [(first + w) % slots for w in range(window)]
            payload = bytes([0xD3]) * total if args.empty else b''.join(source_bytes[index * size:(index + 1) * size] for index in indices)
            assert actual == guard + payload + guard
            assert endpoints[1] > endpoints[0] and sequence.value == 2 * (repeat + 1)
            if repeat >= 2:
                rows.append(dict(first_slot=first, sequence=sequence.value, lcc=list(endpoints), cycles=endpoints[1] - endpoints[0], full_payload_and_guards_correct=True))
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
                if adapter.program_completion(handle).state is bc.EventState.PENDING:
                    adapter.shutdown_program(handle)
                    assert adapter.wait(handle, 15).state is bc.EventState.COMPLETE
            cleanup('stop BC counter', stop)
        if session.value:
            def close() -> None:
                assert shim.MagicHostClose(session) == 0
            cleanup('unmap Host buffer', close)
        if hbm is not None:
            cleanup('free HBM', lambda: adapter.free(handle.device, hbm))
        for device in devices:
            cleanup('destroy wrapper', lambda: adapter.destroy_independent_wrapper(device))
        if installed:
            def remove() -> None:
                code, message = bc.remove_observer(shim)
                if code:
                    raise RuntimeError(message)
            cleanup('remove observer', remove)
        cleanup('clear program', lambda: adapter.clear_program(registration))
        if initialized:
            cleanup('shutdown runtime', adapter.shutdown_process_zero)
    if primary:
        primary.add_note('; '.join(errors))
        raise primary
    assert not errors and not bc.current_process_accel_fds(), errors
    data = {
        'protocol': owner,
        'boundary': 'same-BC LCC sample acknowledged by Host; Host enqueue W native reads and await all callbacks; next same-BC LCC sample',
        'includes_host_bc_marker_handshake': True,
        'counter_words_read_after_end': True,
        'empty_window': args.empty,
        'size_bytes': size,
        'window': window,
        'working_set_bytes': slots * size,
        'seed': args.seed,
        'libtpu': bc.version('libtpu'),
        'counter_device': measured_device,
        'records': rows,
        'cleanup_clean': True,
    }
    (args.output / 'summary.json').write_text(json.dumps(data, indent=2) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=4096)
    parser.add_argument('--window', type=int, choices=(1, 4, 16, 64), default=1)
    parser.add_argument('--empty', action='store_true')
    parser.add_argument('--repeats', type=int, default=24)
    parser.add_argument('--seed', type=int, default=501)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/09_magic_host'))
    args = parser.parse_args()
    if args.size < 4096 or args.size > 4194304 or args.size % 4096 or args.size * args.window > 64 * 1024**2:
        parser.error('每条须为 4 KiB–4 MiB，单窗口至多 64 MiB')
    main(args)
