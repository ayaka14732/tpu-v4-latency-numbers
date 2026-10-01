"""在每颗芯片 TC0 的机器程序中嵌套读取 LCC 与 GTC，用 vdelay 拉长区间，作为第 15 节 BC 比例的同芯片对照。"""

from __future__ import annotations

import argparse
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
import jax.numpy as jnp
from jaxlib.xla_client import LoadedExecutable
import numpy as np

from tpuasm import BundleInsertion, assemble_listing, executable_programs, format_assembly, insert_executable_bundles, load_executable, parse_assembly, replace_executable_programs

TARGET = 'tpu-v4-tc'
PROTOCOL = 'tc-lcc-gtc-vdelay-v1'
# 四个端点依次为外层 BEGIN、内层 BEGIN、内层 END、外层 END；低位寄存器 s20–s23，高位 s25–s28。
LOW = (20, 21, 22, 23)

@pl.kernel(
    out_type=jax.ShapeDtypeStruct((72, 128), jnp.uint32),
    mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
    scratch_types=(pltpu.VMEM((256, 128), jnp.uint32), pltpu.SemaphoreType.DMA),
    name='tc_clock_carrier',
    compiler_params=pltpu.CompilerParams(
        disable_bounds_checks=True,
        disable_semaphore_checks=True,
    ),
)
def kernel(x_hbm: Ref, out_hbm: Ref, data: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, data, sem).wait()
    data[:8, :] = data[:8, :] ^ jnp.uint32(0x13579BDF)
    pltpu.async_copy(data.at[:72, :], out_hbm, sem).wait()

def bundle(instruction: str = '') -> str:
    return '{ ' + instruction + ' }\n'

def read(counter: str, low: int) -> str:
    return bundle(f's0: srdreg.{counter}lo s{low} ; s1: srdreg.{counter}hi s{low + 5}')

def fragment(outer: str, delay: int) -> str:
    inner = 'lcc' if outer == 'gtc' else 'gtc'
    prefix = bundle(f's0: simm.s32 s24, {delay}') + bundle('misc: vnop') * 16 + bundle('s0: sfence')
    body = read(outer, 20) + read(inner, 21) + bundle('misc: vdelay s24') + bundle('s0: sfence') + read(inner, 22) + read(outer, 23)
    suffix = bundle('misc: vnop') * 16
    for tile, register in enumerate((*LOW, *(r + 5 for r in LOW)), 1):
        suffix += bundle(f'va0: vmov.8x128 v12, s{register}') + bundle('misc: vnop') * 8 + bundle(f'vst: vst.8x128 [vmem:0x{tile * 8:x}], v12')
    suffix += bundle('misc: vnop') * 16 + bundle('s0: sfence')
    return f'.target {TARGET}\n' + prefix + body + suffix

def main(args: argparse.Namespace) -> None:
    assert version('libtpu') == '0.0.49'
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    host = rng.integers(0, 1 << 32, (256, 128), dtype=np.uint32)
    expected = host[:8] ^ np.uint32(0x13579BDF)
    rows = []
    for device in jax.local_devices():
        assert device.device_kind == 'TPU v4'
        x = jax.device_put(host, device)
        compiled = jax.jit(kernel, compiler_options={'xla_msa_enable': 'false', 'xla_tpu_vmem_scavenging_mode': 'NONE'}).lower(x).compile()
        np.testing.assert_array_equal(np.asarray(compiled(x))[:8], expected)
        raw = bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize())
        (record, index, image), = executable_programs(raw)
        source = format_assembly(image, target=TARGET)
        program = parse_assembly(source)
        (pc,) = [pc for pc, block in enumerate(program.bundles) for item in block.instructions if item.mnemonic == 'vxor.8x128.u32' and '0x13579bdf' in item.operands]
        local = '\n'.join(str(block.instructions) for block in program.bundles[pc - 1:pc + 4])
        assert not re.search(r'\b(?:s2[0-8]|v12)\b', local), local
        # 插入点在 vxor 之前；vxor 仍对 tile 0 生效，作为载体输出的数值核对。
        for sweep, outers in enumerate((('gtc', 'lcc'), ('lcc', 'gtc'))):
            for outer in outers:
                delays = args.delays if sweep == 0 else args.delays[::-1]
                for delay in delays:
                    text = fragment(outer, delay)
                    patched = insert_executable_bundles(raw, {(record, index): [BundleInsertion(pc, text)]})
                    if device.id == 0 and sweep == 0 and delay == args.delays[-1]:
                        (_, _, full), = executable_programs(patched)
                        listing = format_assembly(full, target=TARGET)
                        assert assemble_listing(listing) == full
                        (args.output / f'{outer}_outer.full.tpuasm').write_text(listing)
                    function = load_executable(patched, compiled)
                    for repeat in range(1 + args.repeats):
                        actual = np.asarray(function(x))
                        np.testing.assert_array_equal(actual[:8], expected)
                        for tile in range(1, 9):
                            assert np.all(actual[tile * 8:(tile + 1) * 8] == actual[tile * 8, 0])
                        halves = [int(actual[tile * 8, 0]) for tile in range(1, 9)]
                        words = [halves[i] | halves[i + 4] << 32 for i in range(4)]
                        if repeat:
                            rows.append(dict(device=device.id, coords=list(device.coords), outer=outer, delay=delay, sweep=sweep, words=words))
        print('device', device.id, device.coords, 'done', flush=True)
    result = dict(protocol=PROTOCOL, libtpu=version('libtpu'), jax=jax.__version__, tpuasm=version('tpuasm'), seed=args.seed, records=rows)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=1) + '\n')
    print('records', len(rows), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delays', type=lambda value: [int(item) for item in value.split(',')], default=[0, 1000, 100000, 10000000, 100000000, 1000000000])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1515)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/15_tc_clock'))
    main(parser.parse_args())
