"""在独立的当前 tpuasm 进程中，为旧 BC runtime 的 TC 源载体加入 CMEM staging。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tpuasm import BundleInsertion, executable_programs, format_assembly, insert_executable_bundles, parse_assembly

def main(args: argparse.Namespace) -> None:
    raw = args.source.read_bytes()
    (record, index, image), = executable_programs(raw)
    listing = format_assembly(image, target='tpu-v4-tc')
    program = parse_assembly(listing)
    operations = [(pc, item) for pc, block in enumerate(program.bundles) for item in block.instructions]
    pointer = f's{args.source_sreg}'
    candidates = [(pc, item) for pc, item in operations if item.mnemonic == 'dma.simple' and item.operands[:3] == (f'[vmem:{pointer}]', '[hbm:s0]', f'length={args.size // 512}')]
    (payload_pc, dma), = candidates
    _, definition = next((pc, item) for pc, item in reversed(operations) if pc < payload_pc and item.operands and item.operands[0] == pointer)
    assert definition.mnemonic == 'simm.s32' and definition.predicate == 15 and int(definition.operands[1], 0) == args.source_row
    ready_pc, _ = next((pc, item) for pc, item in operations if pc > payload_pc and item.mnemonic == 'vsyncadd.remote.s32')
    assert all(not item.operands or item.operands[0] != pointer for pc, item in operations if payload_pc < pc < ready_pc)
    flag = dma.operands[3].removeprefix('dst_flag=')
    rows = args.size // 512
    body = '.target tpu-v4-tc\n{ s0: sfence }\n'
    body += f'{{ s0: dma.simple [cmem:{pointer}], [vmem:{pointer}], length={rows}, dst_flag={flag} }}\n'
    body += f'{{ misc: vwait.ge {flag}, {rows} }}\n{{ misc: vsyncadd.s32 {flag}, -{rows} }}\n{{ s0: sfence }}\n'
    patched = insert_executable_bundles(raw, {(record, index): [BundleInsertion(ready_pc, body)]})
    (_, _, image), = executable_programs(patched)
    args.output.write_bytes(patched)
    args.output.with_suffix('.tpuasm').write_text(format_assembly(image, target='tpu-v4-tc'))
    audit = dict(source_row=args.source_row, source_sreg=args.source_sreg, rows=rows, payload_pc=payload_pc, before_ready_pc=ready_pc, source='vmem', destination='cmem', completion_flag=flag)
    args.output.with_suffix('.json').write_text(json.dumps(audit, indent=2) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sreg', type=int, required=True)
    parser.add_argument('--source-row', type=int, required=True)
    parser.add_argument('--size', type=int, required=True)
    main(parser.parse_args())
