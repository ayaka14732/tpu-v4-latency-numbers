"""在同一 BC 程序中嵌套读取 BC LCC 与 GTC，测量 BC 本地周期相对 GTC 的计数比例。"""

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
sys.path.insert(0, str(BACKEND))

from assembler import default_spec_path, encode_semantic_object, load_spec, parse_source
from backend import AddressUnit, BarnaCoreAllocation, BarnaCoreProgram, BarnaCoreShutdownRequest, EventState, MemorySpace
from jax_allgather_marker import build_shim, current_libtpu_path, install_observer, remove_observer
from libtpu_backend import LibtpuProcessAdapter, SUPPORTED_TARGET
from process_zero_runtime_probe import current_process_accel_fds, RESIDENT_REPLACEMENT_ENVIRONMENT

PROTOCOL = 'bc-lcc-gtc-nested-v1'
# scalar_read_registers 的源 selector：LCC 为 0/1，GTC 为 2/3（low/high）。
SELECTORS = {'lcc': (0, 1), 'gtc': (2, 3)}
MODES = {'spin': 1, 'dma': 2, 'host_wait': 3, 'dense_gtc': 4, 'dense_lcc': 5}
DENSE = ((16, 17), (18, 19), (23, 24), (25, 26), (27, 28), (29, 30))
EMPTY_FENCE = '.s0 scalar_fence c1:v=15 c2:v=1 c4:v=0 a1:v=32 a2:v=255 a3:v=0 a4:v=0 a5:v=0'

def bundle(*instructions: str) -> str:
    return '.bundle\n' + ''.join(line + '\n' for line in instructions) + '.endbundle\n'

def paired(counter: str, low: int, high: int) -> str:
    a, b = SELECTORS[counter]
    return bundle(f'.s0 scalar_read_registers c1:v=15 c2:v=1 c4:v=0 a1:v={a} a2:v={low}', f'.s1 scalar_read_registers c1:v=15 c2:v=1 c4:v=0 a1:v={b} a2:v={high}')

def dma(source: int, destination: int, flag: int, to_hbm: bool) -> str:
    """沿用 bmem_loop.bca 已验证的 compact DMA：flag 清零、done 写回、发起、等待、fence。"""
    direction = 'a10:v=0 a11:v=0 a12:v=0 a13:v=0 a14:v=1 a15:v=4' if to_hbm else 'a10:v=1 a11:v=0 a12:v=0 a13:v=0 a14:v=0 a15:v=0'
    return ''.join([
        bundle(f'.s1 sst [%s{flag}] %s12'),
        bundle(f'.s1 sdone.write %s{flag}, %s12'),
        bundle() * 2,
        bundle(f'.s0 scalar_dma_simple c1:v=15 c2:v=1 c4:v=0 a1:v={source} a2:v=15 a3:v={destination} a4:v={flag} a5:v=0 a6:v=0 a7:v=0 a8:v=0 a9:v=0 {direction} a16:v=0 a17:v=0'),
        bundle(f'.s0 swait.done %s{flag}'),
        bundle(EMPTY_FENCE),
    ])

def counted_loop(label: str, body: str) -> str:
    """s20 从 0 计到 s22；每轮在 body 之后递增并比较，未完成时跳回。"""
    return f'{label}:\n' + body + ''.join([
        bundle('.s0 %s20 = sadd.s32 %s20, %s21'),
        bundle('.s0 %p0 = scmp.eq.s32.totalorder %s20, %s22'),
        bundle(f'.s0 sbr.rel (!%p0) @{label}'),
        bundle(),
    ])

def interval(outer: str, inner: str, work: str) -> str:
    return paired(outer, 16, 17) + paired(inner, 23, 24) + work + paired(inner, 25, 26) + paired(outer, 18, 19)

def source(outer: str) -> str:
    inner = 'lcc' if outer == 'gtc' else 'gtc'
    publish = bundle('.s0 sbr.rel @publish') + bundle()
    paths = {
        'spin': interval(outer, inner, counted_loop('spin_loop', '')) + publish,
        # BMEM 先由 HBM poison 填满；N 次 HBM→BMEM 与最终回读校验均在各自位置，只有 N 次搬运在区间内。
        'dma': dma(4, 12, 13, False) + interval(outer, inner, counted_loop('dma_loop', dma(10, 12, 13, False))) + dma(12, 11, 14, True) + publish,
        'host_wait': interval(outer, inner, bundle('.s0 swait.gt %s9, sy.sel_zero')) + bundle('.s1 sst [%s9] %s12') + publish,
        'dense_gtc': ''.join(paired('gtc', low, high) for low, high in DENSE) + publish,
        'dense_lcc': ''.join(paired('lcc', low, high) for low, high in DENSE) + publish,
    }
    dispatch = ''.join(bundle(f'.s0 %p0 = scmp.eq.s32.totalorder %s7, {mode}') + bundle(f'.s0 sbr.rel (%p0) @{name}') + bundle() for name, mode in MODES.items())
    stores = [f'sst [{7208 + i}] %s{register}' for i, register in enumerate(register for pair in DENSE for register in pair)]
    return ''.join([
        '.target pufferfish-bcs\n',
        f'.sreg_reservation clock_state registers=4-7,9-30 owner={PROTOCOL}\n',
        bundle('.s0 %s5 = smov sy.sel_hex_80000000'),
        'request:\n',
        bundle('.s0 swait.gt %s5, sy.sel_zero'),
        bundle('.s1 scalar_load_smem c1:v=15 c2:v=4 c4:v=0 a1:v=49 a2:v=6 a3:v=0 a4:v=0 a5:v=0 a6:v=0'),
        bundle('.s0 %p0 = scmp.eq.s32.totalorder %s6, 2'),
        bundle('.s0 sbr.rel (%p0) @exit'),
        bundle(),
        bundle('.s0 %s12 = smov sy.sel_zero'),
        bundle('.s1 sst [sy.sel_hex_80000000] %s12'),
        '.scalar_block load_request scoreboard=smem-raw\n',
        '%s10 = sld [barna_core_smem:7200]\n%s11 = sld [barna_core_smem:7201]\n%s22 = sld [barna_core_smem:7202]\n',
        '%s4 = sld [barna_core_smem:7203]\n%s15 = sld [barna_core_smem:7204]\n%s7 = sld [barna_core_smem:7205]\n',
        '%s20 = smov 0\n%s21 = smov 1\n%s13 = smov 8\n%s14 = smov 9\n%s9 = smov 3\n',
        '.endscalar_block\n',
        bundle() * 4,
        dispatch,
        bundle('.s0 sbr.rel @publish'),
        bundle(),
        *(f'{name}:\n{text}' for name, text in paths.items()),
        'publish:\n',
        '.scalar_block publish_counters\n' + '\n'.join(stores) + '\nsst [7220] %s20\n.endscalar_block\n',
        '.scalar_block publish_result\nsst [2] %s21\n.endscalar_block\n',
        bundle('.s0 sbr.rel @request'),
        bundle(),
        'exit:\n',
        bundle('.s0 shalt'),
        '.pad_to 512\n',
    ])

def plan(quick: bool) -> list[dict]:
    jobs = [dict(mode='dense_gtc', count=1, size=4096, wait=0.0), dict(mode='dense_lcc', count=1, size=4096, wait=0.0)]
    spins = [1, 1000, 100000] if quick else [1, 10, 1000, 100000, 1000000, 10000000, 50000000]
    jobs += [dict(mode='spin', count=n, size=4096, wait=0.0) for n in spins]
    for size, counts in ((4096, [1, 100] if quick else [1, 100, 10000, 100000]), (2097152, [1, 4] if quick else [1, 16, 256, 2048])):
        jobs += [dict(mode='dma', count=n, size=size, wait=0.0) for n in counts]
    jobs += [dict(mode='host_wait', count=1, size=4096, wait=t) for t in ([0.0, 0.01] if quick else [0.0, 0.01, 0.1, 0.5, 1.0])]
    return jobs

def main(args: argparse.Namespace) -> None:
    assert 'jax' not in sys.modules and version('libtpu') == SUPPORTED_TARGET.libtpu_version == '0.0.46'
    for name in RESIDENT_REPLACEMENT_ENVIRONMENT:
        os.environ.pop(name, None)
    args.output.mkdir(parents=True, exist_ok=True)
    maximum = 2097152

    def allocation(name: str, space: MemorySpace, address: int) -> BarnaCoreAllocation:
        return BarnaCoreAllocation(name, space, address, AddressUnit.WORD32, 4, 4, PROTOCOL)

    fields = {name: allocation(name, MemorySpace.BARNA_CORE_SMEM, 7200 + i) for i, name in enumerate(('source', 'output', 'count', 'poison', 'size', 'mode'))}
    fields.update({f'half{i}': allocation(f'half{i}', MemorySpace.BARNA_CORE_SMEM, 7208 + i) for i in range(12)})
    fields['done'] = allocation('done', MemorySpace.BARNA_CORE_SMEM, 7220)
    fields.update({name: allocation(name, MemorySpace.BARNA_CORE_SFLAG, index) for name, index in (('wake', 1), ('result', 2), ('release', 3), ('completion', 8), ('return_completion', 9))})
    source_path = args.output / 'program.bca'
    source_path.write_text(source(args.outer))
    spec = load_spec(default_spec_path())
    image = encode_semantic_object(parse_source(source_path, spec), spec, {})
    program = BarnaCoreProgram.from_semantic_proto(image, mailbox_abi=PROTOCOL, required_allocations=tuple(fields.values()), shutdown_request=BarnaCoreShutdownRequest('wake', (2).to_bytes(4, 'little')))
    (args.output / 'program.pb').write_bytes(image)
    shim_path = args.output.parent / 'bc_clock_shim.so'
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
        device = devices[args.device]
        identity = dict(wrapper_id=device.wrapper_id, core_id=device.core_id, chip_coordinates=list(device.chip_coordinates), logical_chip_id=device.logical_chip_id)
        registration, _ = adapter.stage_current_program_shell(program, registration)
        handle = adapter.load_program(device, program)
        # HBM 布局：源、poison、guard、输出、guard；输出区每次请求前重新填 poison。
        hbm = adapter.allocate(device, MemorySpace.HBM, name='clock-source-output', size_bytes=3 * maximum + 1024, alignment_bytes=512)
        base = hbm.byte_address // 512
        rng = np.random.default_rng(args.seed)
        data = rng.integers(0, 256, maximum, dtype=np.uint8).tobytes()
        poison = np.full(maximum // 4, 0xDEADBEEF, dtype=np.uint32).tobytes()
        guard = bytes([0xD3]) * 512
        adapter.write(device, hbm, data + poison + guard + poison + guard)

        def write(name: str, value: int) -> None:
            adapter.write(device, fields[name], int(value).to_bytes(4, 'little'))

        def read(name: str) -> int:
            return int.from_bytes(adapter.read(device, fields[name], size_bytes=4), 'little')

        write('source', base)
        write('poison', base + maximum // 512)
        write('output', base + (2 * maximum + 512) // 512)
        write('wake', 0)
        write('result', 0)
        write('release', 0)
        adapter.start_program(handle)
        jobs = plan(args.quick)
        # 正序与逆序各一轮；每个配置先预热一次，结果按出现顺序全部保存。
        for sweep, ordered in enumerate((jobs, jobs[::-1])):
            for job in ordered:
                for repeat in range(1 + args.repeats):
                    write('mode', MODES[job['mode']])
                    write('count', job['count'])
                    write('size', job['size'] // 512)
                    write('result', 0)
                    if job['mode'] == 'dma':
                        adapter.write(device, hbm, guard + poison[:job['size']] + guard, offset_bytes=2 * maximum)
                    host_begin = time.monotonic_ns()
                    write('wake', 1)
                    if job['mode'] == 'host_wait':
                        time.sleep(job['wait'])
                        write('release', 1)
                    deadline = time.monotonic() + 30
                    while read('result') != 1:
                        if time.monotonic() > deadline:
                            raise TimeoutError(f'BC request did not complete: {job} {adapter.run_states()}')
                    host_end = time.monotonic_ns()
                    halves = [read(f'half{i}') for i in range(12)]
                    words = [halves[i] | halves[i + 1] << 32 for i in range(0, 12, 2)]
                    done = read('done')
                    if job['mode'] == 'dma':
                        actual = adapter.read(device, hbm, offset_bytes=2 * maximum, size_bytes=job['size'] + 1024)
                        assert actual == guard + data[:job['size']] + guard, 'BMEM payload or guard mismatch'
                        assert read('return_completion') == job['size'] // 512
                    if job['mode'] in ('spin', 'dma'):
                        assert done == job['count'], (job, done)
                    if job['mode'] == 'host_wait':
                        assert read('release') == 0
                    if repeat:
                        rows.append(dict(job, sweep=sweep, words=words, done=done, host_ns=host_end - host_begin))
            print('sweep', sweep, 'done', flush=True)
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
    result = dict(protocol=PROTOCOL, libtpu=version('libtpu'), outer=args.outer, device_index=args.device, device=identity, seed=args.seed, records=rows, cleanup_clean=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=1) + '\n')
    print('records', len(rows), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', choices=('gtc', 'lcc'), default='gtc')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1501)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/15_bc_clock/pilot'))
    main(parser.parse_args())
