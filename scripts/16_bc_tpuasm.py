"""在 libtpu 0.0.50 上用 tpuasm 的 BarnaCore 运行时重测第 5、7、9、12 节的 BC 路径。

程序由 tpuasm 汇编（strict），经 `BarnaCoreRuntime` 装载运行；运行器 `bc_tools` 与 TC 交接用的辅助函数取自相邻的 `tpu-v4-barnacore-support`（它的 `pyproject.toml` 固定了 tpuasm 的提交）。除 `clock` 外都在 host 上的第 0 个 BC（芯片 (0,0) 的 BC 0）上计时。子命令：

- `dma`（第 5 节）：HBM→BC 私有 BMEM，W 条 `dma.simple` 背靠背发起，计时区间是一个 BC 程序中 paired LCC 的 BEGIN → W 条发起（每条之后一个 bundle 推进源与目的地址）→ `swait.ge` 等完成 sflag 达到全部 granule 数 → paired END。每个样本之前由 BC 从 HBM poison 区把 BMEM 的 [0, W·S + 1024) 写成 poison，之后把这一段复制到 HBM 的验证区，host 核对 payload 与两侧各 512 B 的 guard。源是 64 MiB 的 HBM 环，每个样本换一个槽，最后三个样本依次是环尾、环头与下一个槽。另测空区间 M 个空 bundle 的校准。
- `pull`（第 7 节）：Megacore Shared CMEM 或 TC VMEM → BMEM，做法同上，源固定。TC VMEM 的源是芯片 (0,0) 上运行中的 Pallas kernel 持有的 2 MiB scratch：kernel 把输入复制进 scratch，把 scratch 的地址与信号量编号交给 BC，等 BC 全部样本结束后用 `remote_scalar_write` 归还的信号量，再把 scratch 复制到输出，host 核对 scratch 没有被改动。
- `host`（第 9 节的对应路径）：host 把 HBM 读进 host，计时用同一个 BC 上的常驻计数程序：host 写 sflag 唤醒 BC 读 BEGIN 并发布序号，host 看到序号后读 HBM，再唤醒 BC 读 END。区间包含两次 host↔BC 握手，空窗口只做握手。读法有两种：`api` 调用 tpuasm 的 `read_hbm`，`native` 把预先分配的缓冲区直接交给它的原生桥接（同步的 `TpuSharedMemory::Read`），用来分开 Python 一侧的开销。
- `clock`（第 12 节）：16 个 BC 各运行两种嵌套顺序（GTC 在外、LCC 在外），区间内为空循环、N 次 HBM→BMEM DMA 与 `swait.gt` 等 host 写 sflag，另有六次连续的 paired LCC 与 paired GTC 读取；四颗芯片的 TC0 在 Pallas kernel 的内联块中做同样的嵌套读取，中间用 `vdelay` 占住 H 个周期。
- `model`：只用 CPU，读取上面各子命令的 JSON，按发现组冻结中心公式（中位数的最小二乘直线），用验证组算残差，并生成 `results/` 中的表。

各设备子命令的 `--phase discovery|validation` 选择尺寸；验证之前先运行 `model --freeze`，冻结的模型写进 `models.json`，之后不允许改写。
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from importlib import import_module
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT.parent / 'tpu-v4-barnacore-support'
sys.path.insert(0, str(SUPPORT))
sys.path.insert(0, str(SUPPORT / 'probes' / '06_sync'))

GRANULE = 512
KIB = 1024
MIB = 1 << 20
WARMUP = 2
SAMPLES = 24
LIST = 4096
RESULTS = 5120
FLAG, POISON_FLAG, VERIFY_FLAG = 20, 21, 22
POISON = 0xa5
RING = 64 * MIB
SCRATCH_ROWS = 4096
GO, SEMAPHORE, SCRATCH = 9, 10, 13
SIZES = {
    'dma': {
        1: ((4, 16, 64, 128, 512, 2048), (8, 32, 96, 256, 1024)),
        4: ((1, 2, 4, 8, 16, 32, 64, 128), (3, 6, 12, 24, 48, 96)),
        16: ((1, 2, 4, 8, 16, 32, 64, 128), (3, 6, 12, 24, 48, 96)),
    },
    'pull': {
        1: ((4, 16, 64, 128, 512, 2048), (8, 32, 96, 256, 1024)),
        4: ((4, 16, 64, 128, 256, 512), (8, 32, 96, 192, 384)),
        16: ((4, 16, 64, 128), (8, 32, 96)),
    },
    'host': {1: ((4, 16, 64, 256, 1024, 4096), (8, 32, 128, 512, 2048))},
}
GAPS = (0, 1, 4, 16, 64)
ENDPOINTS = {'hbm': ('vmem', 'non_core'), 'bmem': ('vmem', 'reserved_self'), 'cmem': ('imem', 'non_core'), 'tc0': ('vmem', 'tensor_core_0')}
MEMORY_IDS = {'vmem': 0, 'imem': 2}
CORE_IDS = {'reserved_self': 0, 'non_core': 1, 'tensor_core_0': 2}
COUNTERS = {'lcc': ('lcc0', 'lcc1'), 'gtc': ('gtc0', 'gtc1')}
SPINS = (1, 10, 1000, 100000, 1000000, 10000000, 50000000)
CLOCK_DMAS = ((8, (1, 100, 10000, 100000)), (4096, (1, 16, 256, 2048)))
WAITS = (0.0, 0.01, 0.1, 0.5, 1.0)
DELAYS = (0, 10, 100, 1000, 10000, 100000, 1000000, 100000000, 1000000000)
TICKS_PER_CYCLE = 32 / 3

def runtimes(count: int) -> list[Any]:
    import bc_tools
    bc_tools.local_chips()
    import jax
    from tpuasm import BarnaCoreRuntime
    jax.local_devices()
    return [BarnaCoreRuntime(core_index=index) for index in range(count)]

def fields(source: str, dest: str) -> str:
    """dma.simple 的端点字段；``outfeed_queue_id`` 与目的端的 core id（高 3 位）、memory id（低 2 位）是同一组位，写成一致的值。"""
    source_memory, source_core = ENDPOINTS[source]
    dest_memory, dest_core = ENDPOINTS[dest]
    queue = CORE_IDS[dest_core] << 2 | MEMORY_IDS[dest_memory]
    return f'source_memory_id={source_memory}, source_core_id={source_core}, src_opcode=read, dma_type_0=local, destination_memory_id={dest_memory}, destination_core_id={dest_core}, outfeed_queue_id={queue}, destination_opcode=write, trace=0'

def simple(source: str, dest: str) -> str:
    """一条 dma.simple：源与目的地址在 s2、s3，长度在 s11，完成 sflag 的编号在 s14。"""
    return f'{{ s0: dma.simple source_address=s2, length=s11, dest_address=s3, dest_sync_flag=s14, {fields(source, dest)} }}'

def reset(flag: int) -> str:
    """s14 设为 flag，并把这个 sflag 的值与 done 位清零（s0 必须为 0）。"""
    return f'{{ s1: smov s14, {flag} }}\n{{ s1: sst [smem:s14], s0 }}\n{{ s1: sdone.write s14, s0 }}\n.empty 8'

def copy(source: str, source_address: str, dest: str, dest_address: str, granules: int, flag: int) -> str:
    """一次不计时的搬运并等它完成（改写 s2、s3、s11、s14）。"""
    return '\n'.join([f'{{ s0: smov s2, {source_address} }}', f'{{ s1: smov s3, {dest_address} }}', f'{{ s0: smov s11, {granules} }}', reset(flag), simple(source, dest), '{ s0: swait.ge s14, s11 }'])

def paired(counter: str, low: int) -> str:
    first, second = COUNTERS[counter]
    return f'{{ s0: srdreg s{low}, {first} ; s1: srdreg s{low + 1}, {second} }}'

def stores(registers: Sequence[int], base: int) -> str:
    return '\n'.join(f'{{ s1: sst [smem:{base + i}], s{register} }}' for i, register in enumerate(registers))

def window_program(source: str, source_address: Callable[[int], str], window: int, granules: int, poison_base: int, verify_base: int) -> str:
    """WARMUP + SAMPLES 个样本，每个样本：poison BMEM → 计时的 W 条搬运 → 复制到 HBM 验证区的第 k 个槽。source_address(k) 给出把第 k 个样本的源地址放进 s2 的 bundle。"""
    span = window * granules + 2
    issues = '\n'.join(f'{simple(source, "bmem")}\n{{ s0: sadd.s32 s2, s2, s11 ; s1: sadd.s32 s3, s3, s11 }}' for _ in range(window))
    blocks = ['{ s0: smov s0, 0 }']
    for k in range(WARMUP + SAMPLES):
        blocks += [
            copy('hbm', str(poison_base), 'bmem', '0', span, POISON_FLAG),
            source_address(k),
            '{ s1: smov s3, 1 }',
            f'{{ s0: smov s11, {granules} }}',
            f'{{ s0: smov s15, {window * granules} }}',
            reset(FLAG),
            paired('lcc', 20),
            issues,
            '{ s0: swait.ge s14, s15 }',
            paired('lcc', 22),
            stores((20, 21, 22, 23), RESULTS + 4 * k),
            copy('bmem', '0', 'hbm', str(verify_base + k * span), span, VERIFY_FLAG),
        ]
    return '\n'.join(blocks)

def check_window(verify: bytes, expected: Sequence[bytes], window: int, size: int) -> list[bool]:
    """每个样本的验证槽：一个 poison granule、W·S 字节的 payload、一个 poison granule。"""
    span = window * size + 2 * GRANULE
    guard = bytes([POISON]) * GRANULE
    return [verify[k * span:k * span + GRANULE] == guard and verify[k * span + GRANULE:k * span + span - GRANULE] == expected[k] and verify[k * span + span - GRANULE:(k + 1) * span] == guard for k in range(WARMUP + SAMPLES)]

def lcc_pairs(runtime: Any) -> list[tuple[int, int]]:
    import bc_tools
    raw = bc_tools.unpack(runtime.read_smem(RESULTS * 4, 16 * (WARMUP + SAMPLES)))
    return [(raw[4 * k] | raw[4 * k + 1] << 32, raw[4 * k + 2] | raw[4 * k + 3] << 32) for k in range(WARMUP + SAMPLES)]

def write_list(runtime: Any, values: Sequence[int]) -> None:
    import bc_tools
    runtime.write_smem(LIST * 4, bc_tools.words(values) + bytes(4096 - 4 * len(values)))

def record(window: int, size: int, pairs: list[tuple[int, int]], correct: list[bool], extra: dict[str, Any]) -> dict[str, Any]:
    rows = [{'lcc': list(pair), 'cycles': pair[1] - pair[0], 'full_payload_and_guards_correct': ok} for pair, ok in zip(pairs, correct)]
    return {'window': window, 'size_bytes': size, 'warmup': rows[:WARMUP], 'records': rows[WARMUP:], **extra}

def ring_slots(window: int, size: int, seed: int) -> list[int]:
    import numpy as np
    slots = RING // (window * size)
    middle = np.random.default_rng(seed).integers(0, slots, WARMUP + SAMPLES - 3).tolist()
    return middle + [slots - 1, 0, 1 % slots]

def dma(arguments: argparse.Namespace) -> None:
    """第 5 节：HBM→BMEM 的大小与窗口矩阵，以及空区间的校准。"""
    import bc_tools
    import numpy as np
    (runtime,) = runtimes(1)
    data = np.random.default_rng(5).integers(0, 256, RING, dtype=np.uint8).tobytes()
    ring = runtime.allocate_hbm(RING)
    runtime.write_hbm(ring, data)
    poison = runtime.allocate_hbm(2 * MIB + 4096)
    runtime.write_hbm(poison, bytes([POISON]) * (2 * MIB + 4096))
    verify = runtime.allocate_hbm(53 * MIB)
    output = {'phase': arguments.phase, 'location': list(runtime.location.chip) + [runtime.location.index], 'configs': [], 'gaps': {}}
    if arguments.phase == 'discovery':
        for gap in GAPS:
            probe = bc_tools.BcsProbe(f'.empty {gap}' if gap else '')
            runtime.load(probe.package())
            output['gaps'][gap] = [probe.run(runtime, load=False).cycles for _ in range(SAMPLES)]
    for window, phases in SIZES['dma'].items():
        for kib in phases[arguments.phase == 'validation']:
            size = kib * KIB
            granules = size // GRANULE
            slots = ring_slots(window, size, seed=window * 100003 + kib)
            write_list(runtime, [ring.byte_address // GRANULE + slot * window * granules for slot in slots])
            program = window_program('hbm', lambda k: f'{{ s1: sld s2, [smem:{LIST + k}] }}', window, granules, poison.byte_address // GRANULE, verify.byte_address // GRANULE)
            bc_tools.BcsProbe(program, clock=False).run(runtime, timeout=120.0)
            span = window * size + 2 * GRANULE
            readback = runtime.read_hbm(verify, -(-(WARMUP + SAMPLES) * span // 4096) * 4096)
            expected = [data[slot * window * size:(slot + 1) * window * size] for slot in slots]
            entry = record(window, size, lcc_pairs(runtime), check_window(readback, expected, window, size), {'slots': slots})
            output['configs'].append(entry)
            print(f'dma W={window} S={size} median={np.median([row["cycles"] for row in entry["records"]])} ok={all(row["full_payload_and_guards_correct"] for row in entry["records"])}', flush=True)
    save(arguments, f'dma_{arguments.phase}', output)
    runtime.close()

def save(arguments: argparse.Namespace, name: str, output: dict[str, Any]) -> None:
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / f'{name}.json').write_text(json.dumps(output) + '\n')
    print(f'saved {arguments.output / name}.json', flush=True)

def tc_source_kernel(chip_id: int) -> Callable[[Any], Any]:
    """TC0 把输入复制进 2 MiB 的 scratch，把 scratch 的 {ref}（以 512 B 计）与信号量编号写进 BC 0 的 sflag 13、10，把 sflag 9 写成 1，等信号量的 1 个 credit，再把 scratch 复制到输出。"""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    import tpuasm
    from sync import remote_line, tc_remote_address
    handover = '\n'.join([remote_line(tc_remote_address(chip_id, 'bc0', SCRATCH), 'vsyncset', '{ref0}'), remote_line(tc_remote_address(chip_id, 'bc0', SEMAPHORE), 'vsyncset', '{ref1}'), remote_line(tc_remote_address(chip_id, 'bc0', GO), 'vsyncset', '1')])

    @pl.kernel(
        out_type=pltpu.HBM((SCRATCH_ROWS, 128), jnp.int32),
        mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
        scratch_types=(pltpu.VMEM((SCRATCH_ROWS, 128), jnp.int32), pltpu.SemaphoreType.REGULAR),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True, skip_device_barrier=True),
    )
    def kernel(x_ref: Any, out_ref: Any, scratch_ref: Any, semaphore: Any) -> None:
        pltpu.sync_copy(x_ref, scratch_ref)
        tpuasm.inline_asm(handover, scratch_ref, semaphore, out_type=(), temporaries={'a': 's'})
        pl.semaphore_wait(semaphore, 1)
        tpuasm.inline_asm('{ s1: sfence }', scratch_ref, out_type=())
        pltpu.sync_copy(scratch_ref, out_ref)

    return jax.jit(kernel, compiler_options={'xla_msa_enable': False})

def pull(arguments: argparse.Namespace) -> None:
    """第 7 节：Megacore Shared CMEM 与运行中 kernel 的 TC VMEM → BMEM。"""
    import bc_tools
    import jax
    import numpy as np
    import tpuasm
    from tpuasm import FsmDescriptor
    (runtime,) = runtimes(1)
    poison = runtime.allocate_hbm(2 * MIB + 4096)
    runtime.write_hbm(poison, bytes([POISON]) * (2 * MIB + 4096))
    verify = runtime.allocate_hbm(53 * MIB)
    data = np.random.default_rng(7).integers(0, 1 << 31, SCRATCH_ROWS * 128, dtype=np.int32)
    raw = data.tobytes()
    cmem = runtime.allocate_cmem(2 * MIB)
    runtime.write_cmem(cmem, raw)
    chip_id = bc_tools.unpack(runtime.read_smem(8186 * 4, 4))[0]
    device = next(device for device in jax.local_devices() if tuple(device.coords) == runtime.location.chip)
    x = jax.device_put(data.reshape(SCRATCH_ROWS, 128), device)
    with tpuasm.compiler():
        compiled = tc_source_kernel(chip_id).lower(x).compile()
    notify = FsmDescriptor('remote_scalar_write', {'value': 1, 'on_chip_address': 0x8000, 'destination_id': chip_id, 'opcode': 2})
    notify_staging, notify_issue = notify.staging_source(temporary='s10', scratch='s13', smem_word_address=16400, packed=True, registers={'on_chip_address': 's8'}).rsplit('\n', 1)
    output = {'phase': arguments.phase, 'location': list(runtime.location.chip) + [runtime.location.index], 'configs': []}
    for source in ('cmem', 'tc0'):
        for window, phases in SIZES['pull'].items():
            for kib in phases[arguments.phase == 'validation']:
                size = kib * KIB
                granules = size // GRANULE
                base = cmem.byte_address // GRANULE
                address: Callable[[int], str] = (lambda k: f'{{ s0: smov s2, {base} }}') if source == 'cmem' else (lambda k: '{ s0: sadd.s32 s2, s7, 0 }')
                body = window_program(source, address, window, granules, poison.byte_address // GRANULE, verify.byte_address // GRANULE)
                if source == 'tc0':
                    # Wait for the kernel's handover (semaphore in s8, scratch in s7), then release the semaphore after all samples.
                    body = '\n'.join([
                        f'{{ s0: smov s1, {GO} ; s1: smov s4, 1 }}',
                        '{ s0: swait.ge s1, s4 }',
                        f'{{ s1: sld s8, [smem:{SEMAPHORE}] }}',
                        f'{{ s1: sld s7, [smem:{SCRATCH}] }}',
                        '.empty 8',
                        body,
                        '{ s0: smov s0, 0 }',
                        notify_staging,
                        notify_issue,
                        '{ s0: sdelay 0xff }',
                    ])
                runtime.load(bc_tools.BcsProbe(body, clock=False).package())
                for flag in (GO, SEMAPHORE, SCRATCH, bc_tools.DONE_FLAG):
                    runtime.write_sflag(flag, 0)
                runtime.start()
                scratch_intact = None
                if source == 'tc0':
                    scratch_intact = bool(np.array_equal(np.asarray(compiled(x)), data.reshape(SCRATCH_ROWS, 128)))
                finished = bc_tools.watch(runtime, timeout=120.0)
                span = window * size + 2 * GRANULE
                readback = runtime.read_hbm(verify, -(-(WARMUP + SAMPLES) * span // 4096) * 4096)
                entry = record(window, size, lcc_pairs(runtime), check_window(readback, [raw[:window * size]] * (WARMUP + SAMPLES), window, size), {'source': source, 'finished': finished, 'scratch_intact': scratch_intact})
                output['configs'].append(entry)
                print(f'pull {source} W={window} S={size} median={np.median([row["cycles"] for row in entry["records"]])} ok={finished and all(row["full_payload_and_guards_correct"] for row in entry["records"])} scratch={scratch_intact}', flush=True)
    save(arguments, f'pull_{arguments.phase}', output)
    runtime.close()

def counter_program() -> str:
    """常驻计数程序：每个样本等 sflag 1、读 BEGIN、清 sflag 1、把序号写进 sflag 2；再等 sflag 1、读 END、清 sflag 1、存读数、把序号写进 sflag 3。"""
    blocks = ['{ s0: smov s0, 0 ; s1: smov s1, 1 }']
    for k in range(WARMUP + SAMPLES):
        blocks += [
            '{ s0: swait.gt s1, 0 }',
            paired('lcc', 20),
            '{ s1: sst [smem:s1], s0 }',
            f'{{ s0: smov s5, {k + 1} }}',
            '{ s1: sst [smem:2], s5 }',
            '{ s0: swait.gt s1, 0 }',
            paired('lcc', 22),
            '{ s1: sst [smem:s1], s0 }',
            stores((20, 21, 22, 23), RESULTS + 4 * k),
            '{ s1: sst [smem:3], s5 }',
        ]
    return '\n'.join(blocks)

def poll(runtime: Any, flag: int, value: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while runtime.read_sflag(flag) != value:
        if time.monotonic() > deadline:
            raise TimeoutError(f'sflag {flag} did not reach {value}')

def host(arguments: argparse.Namespace) -> None:
    """第 9 节的对应路径：host 读 HBM，同一个 BC 的 LCC 包围握手与读取；size 0 是空窗口。``api`` 调用 ``read_hbm``（含它分配缓冲区与复制成 bytes 的 Python 开销），``native`` 把预先分配的缓冲区直接交给同一个原生桥接调用；数据都在 END 之后核对。"""
    import ctypes
    import bc_tools
    import numpy as np
    (runtime,) = runtimes(1)
    data = np.random.default_rng(9).integers(0, 256, RING, dtype=np.uint8).tobytes()
    ring = runtime.allocate_hbm(RING)
    runtime.write_hbm(ring, data)
    probe = bc_tools.BcsProbe(counter_program(), clock=False)
    runtime.load(probe.package())
    output = {'phase': arguments.phase, 'location': list(runtime.location.chip) + [runtime.location.index], 'configs': []}
    kibs = SIZES['host'][1][arguments.phase == 'validation']
    buffer = ctypes.create_string_buffer(4 * MIB)
    for mode in ('api', 'native'):
        for kib in ((0,) if arguments.phase == 'discovery' and mode == 'api' else ()) + kibs:
            size = kib * KIB
            slots = ring_slots(1, size, seed=900001 + kib) if size else [0] * (WARMUP + SAMPLES)
            for flag in (1, 2, 3, bc_tools.DONE_FLAG):
                runtime.write_sflag(flag, 0)
            runtime.start()
            correct = []
            for k, slot in enumerate(slots):
                runtime.write_sflag(1, 1)
                poll(runtime, 2, k + 1)
                received = b''
                if size and mode == 'api':
                    received = runtime.read_hbm(ring, size, byte_offset=slot * size)
                elif size:
                    runtime._call('transfer', 0, False, ring.byte_address + slot * size, buffer, size, -1)
                runtime.write_sflag(1, 1)
                poll(runtime, 3, k + 1)
                # Check after END and drop the bytes, so that the next read_hbm reuses freed host memory as a caller in a loop would.
                correct.append(not size or (received if mode == 'api' else buffer.raw[:size]) == data[slot * size:(slot + 1) * size])
                del received
            finished = bc_tools.watch(runtime, timeout=10.0)
            entry = record(1, size, lcc_pairs(runtime), correct, {'mode': mode, 'slots': slots, 'finished': finished})
            output['configs'].append(entry)
            print(f'host {mode} S={size} median={np.median([row["cycles"] for row in entry["records"]])} ok={finished and all(correct)}', flush=True)
    save(arguments, f'host_{arguments.phase}', output)
    runtime.close()

def interval(outer: str, work: str) -> str:
    """外层 BEGIN（s24/s25）→ 内层 BEGIN（s26/s27）→ work → 内层 END（s16/s17）→ 外层 END（s18/s19），四个端点在相邻的 bundle 中。"""
    inner = 'lcc' if outer == 'gtc' else 'gtc'
    return '\n'.join([paired(outer, 24), paired(inner, 26), work, paired(inner, 16), paired(outer, 18)])

def loop(label: str, body: str) -> str:
    """s20 从 0 计到 s22，每轮 body 之后 4 个 bundle：加 1、比较、分支与它的延迟槽。"""
    return '\n'.join([f'{label}:', body, '{ s0: sadd.s32 s20, s20, 1 }', '{ s0: scmp.lt.s32 p0, s20, s22 }', f'{{ s0: @p0 sbr.rel {label} }}', '{}']).replace(f'{label}:\n\n', f'{label}:\n')

def clock_program(mode: str, outer: str, count: int, granules: int, source: int, poison: int, verify: int) -> str:
    """一个计时区间（或一组连续读取）；端点与循环次数存到 RESULTS 起的字。"""
    endpoints = stores((24, 25, 26, 27, 16, 17, 18, 19, 20), RESULTS)
    start = f'{{ s0: smov s0, 0 ; s1: smov s20, 0 }}\n{{ s0: smov s22, {count} }}'
    if mode == 'spin':
        return '\n'.join([start, '.empty 4', interval(outer, loop('spin', '')), endpoints])
    if mode == 'dma':
        work = loop('dma', '\n'.join([reset(FLAG), simple('hbm', 'bmem'), '{ s0: swait.ge s14, s11 }']))
        setup = '\n'.join([f'{{ s0: smov s2, {source} }}', '{ s1: smov s3, 1 }', f'{{ s0: smov s11, {granules} }}', '.empty 4'])
        return '\n'.join([start, copy('hbm', str(poison), 'bmem', '0', granules + 2, POISON_FLAG), setup, interval(outer, work), endpoints, copy('bmem', '0', 'hbm', str(verify), granules + 2, VERIFY_FLAG)])
    if mode == 'host_wait':
        return '\n'.join([start, '{ s0: smov s9, 3 ; s1: smov s5, 1 }', '{ s1: sst [smem:4], s5 }', '.empty 4', interval(outer, '{ s0: swait.gt s9, 0 }'), '{ s1: sst [smem:s9], s0 }', endpoints])
    counter = mode.split('_')[1]
    reads = [paired(counter, low) for low in (16, 18, 20, 22, 24, 26)]
    return '\n'.join(['{ s0: smov s0, 0 }', '.empty 4', *reads, stores(range(16, 28), RESULTS)])

def clock_plan() -> list[tuple[str, int, int, float]]:
    """(mode, count, granules, host 等待秒数)。"""
    plan = [('dense_gtc', 0, 0, 0.0), ('dense_lcc', 0, 0, 0.0)]
    plan += [('spin', count, 0, 0.0) for count in SPINS]
    plan += [('dma', count, granules, 0.0) for granules, counts in CLOCK_DMAS for count in counts]
    plan += [('host_wait', 1, 0, wait) for wait in WAITS]
    return plan

def tc_clock_kernel(outer: str) -> Callable[[Any], Any]:
    """TC0 的嵌套读取：sfence → 外层 BEGIN → 内层 BEGIN → vdelay H → sfence → 内层 END → 外层 END；H 从 SMEM 输入读入。"""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    import tpuasm
    inner = 'lcc' if outer == 'gtc' else 'gtc'

    def read(counter: str, index: int) -> str:
        return f'{{ s0: srdreg.{counter}lo {{dst{index}}} ; s1: srdreg.{counter}hi {{dst{index + 4}}} }}'

    template = '\n'.join(['{ s0: sfence }', read(outer, 0), read(inner, 1), '{ misc: vdelay {src0} }', '{ s0: sfence }', read(inner, 2), read(outer, 3)])

    @pl.kernel(
        out_type=pltpu.SMEM((8,), jnp.int32),
        mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True, skip_device_barrier=True, has_side_effects=True),
    )
    def kernel(h_ref: Any, out_ref: Any) -> None:
        values = tpuasm.inline_asm(template, h_ref[0], out_type=tuple(jax.ShapeDtypeStruct((), jnp.int32) for _ in range(8)), clobbers=('clock',))
        for i, value in enumerate(values):
            out_ref[i] = value

    def call(h: Any) -> Any:
        return kernel(pltpu.with_memory_space_constraint(h, pltpu.SMEM))
    return jax.jit(call)

def clock(arguments: argparse.Namespace) -> None:
    """第 12 节：16 个 BC 与四颗芯片 TC0 的 LCC/GTC 嵌套读取。"""
    import threading
    import bc_tools
    import jax
    import numpy as np
    import tpuasm
    cores = runtimes(16)
    output: dict[str, Any] = {'bc': [], 'tc': []}
    devices = {tuple(device.coords): device for device in jax.local_devices()}
    for outer in ('gtc', 'lcc'):
        with tpuasm.compiler():
            compiled = {chip: tc_clock_kernel(outer).lower(jax.device_put(np.zeros((1,), np.int32), device)).compile() for chip, device in sorted(devices.items())}
        for chip, device in sorted(devices.items()):
            rows = []
            for delay in (DELAYS[0],) + DELAYS:
                values = np.asarray(compiled[chip](jax.device_put(np.array([delay], np.int32), device))).astype(np.uint32).tolist()
                rows.append({'delay': delay, 'words': values})
            output['tc'].append({'chip': list(chip), 'outer': outer, 'samples': rows})
            print(f'tc chip {chip} outer {outer}: {len(rows)} samples', flush=True)
    save(arguments, 'clock', output)
    plan = clock_plan()
    for runtime in cores:
        ring = runtime.allocate_hbm(4096 * GRANULE)
        data = np.random.default_rng(runtime.core_index).integers(0, 256, 4096 * GRANULE, dtype=np.uint8).tobytes()
        runtime.write_hbm(ring, data)
        poison = runtime.allocate_hbm(4098 * GRANULE + 3072)
        runtime.write_hbm(poison, bytes([POISON]) * (4098 * GRANULE + 3072))
        verify = runtime.allocate_hbm(4098 * GRANULE + 3072)
        for outer in ('gtc', 'lcc'):
            bases = (ring.byte_address // GRANULE, poison.byte_address // GRANULE, verify.byte_address // GRANULE)
            programs = {index: bc_tools.BcsProbe(clock_program(mode, outer, count, granules, *bases), clock=False).package() for index, (mode, count, granules, _) in enumerate(plan)}
            passes = [('warmup', list(range(len(plan))))] + [('forward', list(range(len(plan))))] * 3 + [('reverse', list(reversed(range(len(plan)))))] * 3
            samples = []
            for name, order in passes:
                for index in order:
                    mode, count, granules, wait = plan[index]
                    runtime.load(programs[index])
                    for flag in (3, 4, bc_tools.DONE_FLAG):
                        runtime.write_sflag(flag, 0)
                    waker = None
                    if mode == 'host_wait':
                        def wake(seconds: float = wait) -> None:
                            poll(runtime, 4, 1)
                            time.sleep(seconds)
                            runtime.write_sflag(3, 1)
                        waker = threading.Thread(target=wake)
                    began = time.monotonic_ns()
                    runtime.start()
                    if waker is not None:
                        waker.start()
                    finished = bc_tools.watch(runtime, timeout=30.0)
                    elapsed = time.monotonic_ns() - began
                    if waker is not None:
                        waker.join()
                    words = bc_tools.unpack(runtime.read_smem(RESULTS * 4, 4 * 12))
                    sample = {'pass': name, 'mode': mode, 'count': count, 'granules': granules, 'wait': wait, 'finished': finished, 'host_ns': elapsed, 'words': list(words)}
                    if mode == 'dma':
                        readback = runtime.read_hbm(verify, -(-(granules + 2) * GRANULE // 4096) * 4096)
                        guard = bytes([POISON]) * GRANULE
                        sample['payload_and_guards_correct'] = readback[:GRANULE] == guard and readback[GRANULE:(granules + 1) * GRANULE] == data[:granules * GRANULE] and readback[(granules + 1) * GRANULE:(granules + 2) * GRANULE] == guard
                    samples.append(sample)
            output['bc'].append({'core_index': runtime.core_index, 'chip': list(runtime.location.chip), 'core': runtime.location.index, 'outer': outer, 'samples': samples})
            print(f'clock core {runtime.core_index} outer {outer}: {len(samples)} samples, all finished {all(s["finished"] for s in samples)}', flush=True)
    save(arguments, 'clock', output)
    for runtime in cores:
        runtime.close()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument('command', choices=('dma', 'pull', 'host', 'clock', 'model'))
    parser.add_argument('--phase', choices=('discovery', 'validation'), default='discovery')
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence/16_bc_tpuasm'))
    parser.add_argument('--freeze', action='store_true', help='model: freeze the discovery models into models.json')
    arguments = parser.parse_args()
    if arguments.command == 'model':
        import_module('16_bc_tpuasm_model').main(arguments)
        return
    {'dma': dma, 'pull': pull, 'host': host, 'clock': clock}[arguments.command](arguments)

if __name__ == '__main__':
    main()
