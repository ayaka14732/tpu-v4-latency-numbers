"""仅用 CPU 从实际提交的 BC semantic program 重建代表性机器清单，记录两个 libtpu 版本。"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tpuasm' / 'src'))

from tpuasm import assemble_listing, encode_tpu_v4_bcs_program, format_assembly

def main(args: argparse.Namespace) -> None:
    assert version('libtpu') == '0.0.49', '清单重建使用当前 0.0.49 codec，不加载原生设备 runtime'
    total = 0
    for family in args.families:
        assert family in ('05_bc', '07_bc_pull')
        destination = args.archive / family
        destination.mkdir(parents=True, exist_ok=True)
        cases = [(None, window) for window in (1, 4, 16)] if family == '05_bc' else [(space, window) for space in ('vmem', 'cmem') for window in (1, 4, 16)]
        programs = []
        for space, window in cases:
            name = f'w{window}_4k' if space is None else f'{space}_w{window}_4k'
            source = args.root / ('05_bc_suite' if space is None else '07_bc_suite')
            source /= f'discovery_w{window}_s4096' if space is None else f'{space}_discovery_w{window}_s4096'
            data = json.loads((source / 'summary.json').read_text())
            assert data['libtpu'] == '0.0.46' and data['size_bytes'] == 4096 and data['window'] == window
            assert len(data['records']) == 24
            semantic = (source / 'program.pb').read_bytes()
            # 此 API 检查显式 semantic 字段未被 codec 消隐，并核对编码往返；不提交程序到设备。
            image = encode_tpu_v4_bcs_program(semantic)
            listing = format_assembly(image, target='tpu-v4-bcs')
            assert assemble_listing(listing) == image
            output = destination / f'{name}.tpuasm'
            output.write_text(listing)
            programs.append(dict(name=name, semantic_source=str(source / 'program.pb'), image_bytes=len(image), exact_listing_roundtrip=True))
            total += 1
            print('verified', output, flush=True)
        metadata = dict(
            semantic_program_runtime_libtpu='0.0.46',
            listing_codec_libtpu='0.0.49',
            note='由实际提交的 semantic program 经当前 codec 编码，逐字段验证并通过 codec 往返；不是旧 runtime 的 LLO dump。',
            programs=programs,
        )
        (destination / 'listing_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    print(total, 'BC listings reconstructed and checked on CPU')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/tmp/tpu_latency_numbers'))
    parser.add_argument('--archive', type=Path, default=Path('/tmp/tpu_latency_numbers/evidence'))
    parser.add_argument('--families', type=lambda value: value.split(','), default=['05_bc', '07_bc_pull'])
    main(parser.parse_args())
