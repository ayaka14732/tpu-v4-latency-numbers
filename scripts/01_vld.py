"""用 tpuasm 在 TPU v4 机器程序中插入探针，测量 vld.8x128 的 load-use 间隔、发射吞吐与 fence 尾部。

运行：/srv/workspace/venv/bin/python scripts/01_vld.py
汇编、原始读数和数值结果默认写入 /tmp/tpu_latency_numbers/01_vld。
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
SENTINEL = 0xDEADBEEF

@pl.kernel(
    out_type=jax.ShapeDtypeStruct((56, 128), jnp.uint32),
    mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
    scratch_types=(pltpu.VMEM((256, 128), jnp.uint32), pltpu.SemaphoreType.DMA),
    name='vld_tpuasm_carrier',
    compiler_params=pltpu.CompilerParams(
        disable_bounds_checks=True,
        disable_semaphore_checks=True,
    ),
)
def kernel(x_hbm: Ref, out_hbm: Ref, data: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, data, sem).wait()
    data[:8, :] = data[:8, :] ^ jnp.uint32(0x13579BDF)
    pltpu.async_copy(data.at[:56, :], out_hbm, sem).wait()

def bundle(instruction: str = '') -> str:
    return '{ ' + instruction + ' }\n'

def gap(kind: str, count: int) -> str:
    instruction = {'empty': '', 'scalar': 's0: sadd.s32 s24, 0, s24', 'vnop': 'misc: vnop', 'delay0': 'misc: vdelay 0', 'delay1': 'misc: vdelay 1'}[kind]
    return bundle(instruction) * count

def read(register: int) -> str:
    # 同一 bundle 的 S0/S1 读取 LCC 低、高 32 位，组成一致的 64 位快照；高位放在 s(register + 5)。
    return bundle(f's0: srdreg.lcclo s{register} ; s1: srdreg.lcchi s{register + 5}')

class Probe:
    def __init__(self, output: Path, repeats: int, full_listing_names: tuple[str, ...] = ()) -> None:
        self.output = output
        self.repeats = repeats
        self.full_listing_names = full_listing_names
        output.mkdir(parents=True, exist_ok=True)
        (output / 'results.jsonl').write_text('')
        device = jax.local_devices()[0]
        assert device.device_kind == 'TPU v4'
        assert version('libtpu') == '0.0.49'
        rng = np.random.default_rng(20260928)
        self.host = rng.integers(0, 1 << 32, (256, 128), dtype=np.uint32)
        self.x = jax.device_put(self.host, device)
        self.compiled = jax.jit(kernel, compiler_options={'xla_msa_enable': 'false', 'xla_tpu_vmem_scavenging_mode': 'NONE'}).lower(self.x).compile()
        self.baseline()
        raw = bytes(cast(LoadedExecutable, self.compiled.runtime_executable()).serialize())
        (record, index, image), = executable_programs(raw)
        self.key = record, index
        source = format_assembly(image, target=TARGET)
        (output / 'carrier.tpuasm').write_text(source)
        program = parse_assembly(source)
        markers = [(pc, item) for pc, block in enumerate(program.bundles) for item in block.instructions if item.mnemonic == 'vxor.8x128.u32' and '0x13579bdf' in item.operands]
        (self.pc, marker), = markers
        assert len(program.bundles[self.pc].instructions) == 1
        # 此载体的输入与输出使用同一 scratch；核对真实地址，不把分配位置当成 ABI。
        print(f'carrier marker PC={self.pc:#x}, output register={marker.operands[0]}', flush=True)
        loads = [item for block in program.bundles[max(0, self.pc - 8):self.pc] for item in block.instructions if item.mnemonic == 'vld.8x128']
        assert len(loads) == 1 and loads[0].operands[1] == '[vmem:0x0]', loads
        local = '\n'.join(str(block.instructions) for block in program.bundles[self.pc - 1:self.pc + 4])
        assert not re.search(r'\b(?:s2[0-7]|v1[0-4])\b', local), local
        old = f'{marker.slot}: {marker.mnemonic} ' + ', '.join(marker.operands)
        assert source.count(old) == 1
        source = source.replace(old, f'{marker.slot}: vmov.8x128 {marker.operands[0]}, v10')
        self.raw = replace_executable_programs(raw, {self.key: assemble_listing(source)})
        backend, _ = select_backend(TARGET)
        self.environment = {
            'python': sys.version,
            'jax': jax.__version__,
            'jaxlib': version('jaxlib'),
            'libtpu': version('libtpu'),
            'libtpu_build_id': backend.build_id,
            'tpuasm': version('tpuasm'),
            'device': str(device),
            'device_kind': device.device_kind,
            'repeats': repeats,
            'seed': 20260928,
        }
        (output / 'environment.json').write_text(json.dumps(self.environment, indent=2) + '\n')
        np.save(output / 'input.npy', self.host)
        self.records: list[dict] = []

    def baseline(self) -> None:
        expected = self.host[:56].copy()
        expected[:8] ^= np.uint32(0x13579BDF)
        np.testing.assert_array_equal(np.asarray(self.compiled(self.x)), expected)

    def run(self, name: str, body: str, expected: np.ndarray | None = None, setup: str = '') -> dict:
        # v11 的旧值和 v10 的初值在计时前建立；sfence 隔离输入 DMA 与初始化。
        prefix = bundle('va0: vimm.8x128.s32 v11, -559038737') + bundle('vld: vld.8x128 v10, [vmem:0x0]')
        prefix += bundle('s0: simm.s32 s24, 0') + gap('vnop', 16) + setup + bundle('s0: sfence')
        # 三个端点的 LCC 低、高位各经独立的 8×128 tile 返回；所有输出写入均在计时后。
        suffix = gap('vnop', 16)
        for tile, register in enumerate((20, 21, 22, 25, 26, 27), 1):
            suffix += bundle(f'va0: vmov.8x128 v12, s{register}') + gap('vnop', 8)
            suffix += bundle(f'vst: vst.8x128 [vmem:0x{tile * 8:x}], v12')
        suffix += gap('vnop', 16) + bundle('s0: sfence')
        fragment = f'.target {TARGET}\n' + prefix + body + suffix
        patched = insert_executable_bundles(self.raw, {self.key: [BundleInsertion(self.pc, fragment)]})
        (self.output / f'{name}.tpuasm').write_text(fragment)
        if name in ('consumer-add0-distance0', 'consumer-add0-distance1', 'stream-dependent-128', 'tail-vld-32-f1', *self.full_listing_names):
            (_, _, image), = executable_programs(patched)
            listing = format_assembly(image, target=TARGET)
            assert assemble_listing(listing) == image
            (self.output / f'{name}.full.tpuasm').write_text(listing)
        function = load_executable(patched, self.compiled)
        values = []
        for _ in range(self.repeats):
            actual = np.asarray(function(self.x))
            for tile in range(1, 7):
                assert np.all(actual[tile * 8:(tile + 1) * 8] == actual[tile * 8, 0])
            values.append(actual)
        outputs = np.stack(values)
        np.save(self.output / f'{name}.npy', outputs)
        halves = outputs[:, (8, 16, 24, 32, 40, 48), 0].astype(np.uint64)
        counters = halves[:, :3] | (halves[:, 3:] << np.uint64(32))
        differences = counters[:, 1:] - counters[:, :1]
        if expected is None:
            expected = self.host[:8]
        matches = np.all(outputs[:, :8] == expected, axis=(1, 2))
        record = {
            'name': name,
            'lcc': counters.tolist(),
            'gaps_from_begin': differences.tolist(),
            'correct': matches.tolist(),
            'matching_words': np.sum(outputs[:, :8] == expected, axis=(1, 2)).tolist(),
            'sentinel_words': np.sum(outputs[:, :8] == SENTINEL, axis=(1, 2)).tolist(),
        }
        self.records.append(record)
        with (self.output / 'results.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(name, 'gaps', np.unique(differences, axis=0).tolist(), 'correct', int(np.sum(matches)), '/', self.repeats, 'words', sorted(set(record['matching_words'])), flush=True)
        return record

LOAD = 'vld: vld.8x128 v11, [vmem:0x0]'
# 正控制：v9 = 1.5f，vmul 得到 2.25f。它的结果晚于下一 bundle 才就绪，由硬件互锁补足等待。
MUL = 'va0: vmul.8x128.f32 v11, v9, v9'
MUL_SETUP = bundle('va0: vimm.8x128.s32 v9, 1069547520') + gap('vnop', 16)
MUL_INPUT, MUL_RESULT = 0x3FC00000, 0x40100000

def consumers(probe: Probe) -> None:
    # consumer 与 load 相距 d 个 bundle；d=0 应读到计时前写入的旧值。
    for consumer in ('add0', 'add1', 'move', 'store'):
        instruction = {
            'add0': 'va0: vadd.8x128.s32 v10, 1, v11',
            'add1': 'va1: vadd.8x128.s32 v10, 1, v11',
            'move': 'va0: vmov.8x128 v10, v11',
            'store': 'vst: vst.8x128 [vmem:0x80], v11',
        }[consumer]
        for distance in (0, 1, 2, 4):
            work = bundle(LOAD + ' ; ' + instruction) if distance == 0 else bundle(LOAD) + gap('vnop', distance - 1) + bundle(instruction)
            body = read(20) + work + read(21) + bundle('s0: sfence') + read(22)
            if consumer == 'store':
                body += gap('vnop', 16) + bundle('vld: vld.8x128 v10, [vmem:0x80]')
            expected = np.full((8, 128), SENTINEL, np.uint32) if distance == 0 else probe.host[:8]
            if consumer.startswith('add'):
                expected = expected + np.uint32(1)
            result = probe.run(f'consumer-{consumer}-distance{distance}', body, expected)
            assert all(result['correct'])
    # vmul 在 d=1 同样读到新值：数值正确只说明有 forwarding 或互锁，不说明间隔为 1 cycle。
    for distance in (0, 1, 2):
        consumer = 'va1: vadd.8x128.s32 v10, 1, v11'
        work = bundle(MUL + ' ; ' + consumer) if distance == 0 else bundle(MUL) + gap('vnop', distance - 1) + bundle(consumer)
        body = read(20) + work + read(21) + bundle('s0: sfence') + read(22)
        expected = np.full((8, 128), (SENTINEL if distance == 0 else MUL_RESULT) + 1, np.uint32)
        result = probe.run(f'consumer-mul-distance{distance}', body, expected, MUL_SETUP)
        assert all(result['correct'])

def streams(probe: Probe) -> None:
    # 长序列的斜率给出稳态吞吐与互锁等待；dependent 与 independent 只差 consumer 读的是刚写入的 v11 还是预先准备的 v9。
    for kind in ('independent', 'dependent', 'mul-independent', 'mul-dependent', 'vld', 'vnop', 'empty', 'delay1'):
        for count in (16, 64, 128):
            setup = bundle('vld: vld.8x128 v9, [vmem:0x0]') + gap('vnop', 16)
            expected = probe.host[:8]
            source = 'v11' if kind.endswith('-dependent') or kind == 'dependent' else 'v9'
            if kind in ('independent', 'dependent'):
                work = (bundle(LOAD) + bundle(f'va0: vadd.8x128.s32 v10, 1, {source}')) * count
                expected = probe.host[:8] + np.uint32(1)
            elif kind.startswith('mul-'):
                work = (bundle(MUL) + bundle(f'va0: vadd.8x128.s32 v10, 1, {source}')) * count
                setup = MUL_SETUP
                expected = np.full((8, 128), (MUL_RESULT if source == 'v11' else MUL_INPUT) + 1, np.uint32)
            elif kind == 'vld':
                work = ''.join(bundle(f'vld: vld.8x128 v{i % 8}, [vmem:0x{(i % 32) * 8:x}]') for i in range(count))
            else:
                work = gap(kind, count)
            body = read(20) + work + read(21) + bundle('s0: sfence') + read(22)
            result = probe.run(f'stream-{kind}-{count}', body, expected, setup)
            assert all(result['correct'])

def fence_tails(probe: Probe) -> None:
    # 空区间中 fence 的基线。
    for count in (0, 1, 2, 4):
        result = probe.run(f'fence-only-{count}', read(20) + bundle('s0: sfence') * count + read(21) + read(22))
        assert all(result['correct'])
    # N 条连续 load 后有无 fence；N>=11 会覆盖 v10，计时后恢复返回数据。
    for count in (1, 2, 4, 16, 32):
        loads = ''.join(bundle(f'vld: vld.8x128 v{i}, [vmem:0x{i * 8:x}]') for i in range(count))
        for fence in (False, True):
            body = read(20) + loads + (bundle('s0: sfence') if fence else '') + read(21) + read(22)
            body += bundle('vld: vld.8x128 v10, [vmem:0x0]')
            result = probe.run(f'tail-vld-{count}-f{int(fence)}', body)
            assert all(result['correct'])
    # 单条向量指令后接 K 个填充 bundle，再接 fence。
    operations = {
        'load': LOAD,
        'store': 'vst: vst.8x128 [vmem:0x80], v9',
        'imm': 'va0: vimm.8x128.s32 v11, 7',
        'add': 'va0: vadd.8x128.s32 v11, 1, v9',
        'move': 'va0: vmov.8x128 v11, v9',
    }
    setup = bundle('vld: vld.8x128 v9, [vmem:0x0]') + gap('vnop', 16)
    for operation, instruction in operations.items():
        paddings = ('empty', 'scalar', 'vnop', 'delay0', 'delay1') if operation == 'load' else ('empty', 'vnop')
        for padding in paddings:
            for count in (0, 1, 2, 4, 16, 64):
                body = read(20) + bundle(instruction) + gap(padding, count) + bundle('s0: sfence') + read(21) + read(22)
                result = probe.run(f'path-{operation}-{padding}-{count}', body, setup=setup)
                assert all(result['correct'])

def issue_offsets(probe: Probe) -> None:
    # 各类向量指令从标量发射到向量发射的偏移：X 后接 fence、X 与 fence 同 bundle、X 后隔一个空 bundle，以及 64 条连续 X。
    operations = {
        'load': LOAD,
        'load-sreg': 'vld: vld.8x128 v11, [vmem:s24+0x8]',
        'load-stride': 'vld: vld.8x128 v11, [vmem:0x8, ss=2]',
        'load-mask': 'vld: vld.8x128 v11, [vmem:0x8, sm=0xf]',
        'load-shuffle': 'vld: vld.sshfl.8x128 v11, [vmem:0x0], 0',
        'load-imm': LOAD + ' ; va0: vimm.8x128.s32 v12, 1',
        'store': 'vst: vst.8x128 [vmem:0x80], v9',
        'store-sreg': 'vst: vst.8x128 [vmem:s24+0x80], v9',
        'store-mask': 'vst: vst.msk.8x128 [vmem:0x80], vm0, v9',
        'imm': 'va0: vimm.8x128.s32 v11, 7',
        'imm-va1': 'va1: vimm.8x128.s32 v11, 7',
        'add': 'va0: vadd.8x128.s32 v11, 1, v9',
        'add-sreg': 'va0: vadd.8x128.s32 v11, s24, v9',
        'move': 'va0: vmov.8x128 v11, v9',
        'move-sreg': 'va0: vmov.8x128 v11, s24',
        'sync': 'misc: vsyncadd.s32 [sflag:100], 1',
        'vnop': 'misc: vnop',
        'delay1': 'misc: vdelay 1',
        'rotate': 'va0: vrot.slane.down.8x128.u32 v11, v9',
    }
    # 其他槽和其他结果路径：推入 FIFO 的指令在计时后弹出，不做连续 64 条的对照。
    fifo_operations = {
        'v2s-push': ('vst: vpush v2sf, v9', 's0: spop s23, v2sf'),
        'eup-push': ('va0: vpush.8x128 erf, v9', 'vr0: vpop.8x128 v13, erf'),
        'permute': ('vx0: vperm.0.8x128 trf0, v9', 'vr0: vpop.8x128 v13, trf0'),
        'matpush': ('vx0: vmatpush.8x128.f32 gsfn0, v9', ''),
        'cmem-load': ('cld: cld.8x128 crf, [cmem:0x0]', 'vr0: vpop.8x128 v13, crf'),
    }
    setup = bundle('vld: vld.8x128 v9, [vmem:0x0]') + gap('vnop', 16)
    result = probe.run('offset-none-cofence', read(20) + bundle('s0: sfence') + read(21) + read(22), setup=setup)
    assert all(result['correct'])
    for operation, instruction in operations.items():
        bodies = {
            'fence': read(20) + bundle(instruction) + bundle('s0: sfence') + read(21) + read(22),
            'cofence': read(20) + bundle('s0: sfence ; ' + instruction) + read(21) + read(22),
            'empty-fence': read(20) + bundle(instruction) + bundle() + bundle('s0: sfence') + read(21) + read(22),
            'stream': read(20) + bundle(instruction) * 64 + read(21) + bundle('s0: sfence') + read(22),
        }
        for kind, body in bodies.items():
            result = probe.run(f'offset-{operation}-{kind}', body, setup=setup)
            assert all(result['correct'])
    for operation, (instruction, pop) in fifo_operations.items():
        tail = gap('vnop', 32) + (bundle(pop) if pop else '') + gap('vnop', 8)
        bodies = {
            'fence': read(20) + bundle(instruction) + bundle('s0: sfence') + read(21) + read(22),
            'cofence': read(20) + bundle('s0: sfence ; ' + instruction) + read(21) + read(22),
            'empty-fence': read(20) + bundle(instruction) + bundle() + bundle('s0: sfence') + read(21) + read(22),
            'vnop-fence': read(20) + bundle(instruction) + bundle('misc: vnop') + bundle('s0: sfence') + read(21) + read(22),
        }
        for kind, body in bodies.items():
            result = probe.run(f'offset-{operation}-{kind}', body + tail, setup=setup)
            assert all(result['correct'])

def vif(probe: Probe) -> None:
    # 用 vdelay s23 长时间占住向量发射，再在其后排 M 个 bundle；R1 被阻塞时说明标量侧看到 VIF 已满。
    for hold in (200, 1000):
        setup = bundle('vld: vld.8x128 v9, [vmem:0x0]') + bundle(f's0: simm.s32 s23, {hold}') + gap('vnop', 16)
        fillers = {
            'vnop': ('misc: vnop', range(25)),
            'empty': ('', (0, 20, 24)),
            'pair': ('va0: vimm.8x128.s32 v13, 1 ; va1: vimm.8x128.s32 v14, 2', (19, 20, 21)),
            'load': ('vld: vld.8x128 v13, [vmem:0x0]', (19, 20, 21)),
        }
        for filler, (instruction, counts) in fillers.items():
            if hold == 200 and filler != 'vnop':
                continue
            for count in counts:
                body = read(20) + bundle('misc: vdelay s23') + bundle(instruction) * count + read(21) + bundle('s0: sfence') + read(22)
                result = probe.run(f'vif-{hold}-{filler}-{count}', body, setup=setup)
                assert all(result['correct'])
        # R1 与 vnop 同 bundle：进 VIF 的 bundle 与纯标量 bundle 是否在同一占用数下被阻塞。
        for count in (18, 19, 20):
            body = read(20) + bundle('misc: vdelay s23') + bundle('misc: vnop') * count
            body += bundle('s0: srdreg.lcclo s21 ; s1: srdreg.lcchi s26 ; misc: vnop') + bundle('s0: sfence') + read(22)
            result = probe.run(f'vif-{hold}-coread-{count}', body, setup=setup)
            assert all(result['correct'])

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/01_vld'))
    parser.add_argument('--repeats', type=int, default=24)
    groups = {'consumers': consumers, 'streams': streams, 'fence_tails': fence_tails, 'issue_offsets': issue_offsets, 'vif': vif}
    parser.add_argument('--group', choices=(*groups, 'all'), default='all')
    args = parser.parse_args()
    probe = Probe(args.output, args.repeats)
    for name, group in groups.items():
        if args.group in (name, 'all'):
            group(probe)
    probe.baseline()
    (args.output / 'summary.json').write_text(json.dumps({'environment': probe.environment, 'records': probe.records, 'baseline_after': True}, indent=2) + '\n')

if __name__ == '__main__':
    main()
