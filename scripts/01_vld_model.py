"""用 README 第 0 节的发射模型重放 01_vld.py 的每个探针片段，与真机 raw gap 逐一比较。

运行：/srv/workspace/venv/bin/python scripts/01_vld_model.py
默认读取 /tmp/tpu_latency_numbers/01_vld 中的片段与 results.jsonl；不需要 TPU。
"""

import argparse
import json
from pathlib import Path
import re

VIF_ENTRIES = 20  # 标量侧看到 20 项未释放时，任何 bundle 都不能标量发射。
RELEASE = 10  # 一项在向量发射后 10 cycles 才在标量侧释放。
RESULT_LATENCY = {'vmul.8x128.f32': 2}  # 其余 producer 的结果在下一 cycle 可用。

def bundles(path: Path) -> list[list[str]]:
    return [[item.strip() for item in body.split(';') if item.strip()] for body in re.findall(r'\{(.*?)\}', path.read_text())]

def replay(program: list[list[str]]) -> dict[int, int]:
    """返回各 LCC 目的寄存器最后一次读数的标量发射时刻。"""
    scalar = 0  # 当前 bundle 的标量发射时刻
    vector_free = 0  # 向量侧可以发射下一个 bundle 的最早时刻
    releases: list[int] = []  # 尚未在标量侧释放的 VIF 项的释放时刻
    ready: dict[str, int] = {}
    registers: dict[str, int] = {}
    reads = {}
    for items in program:
        pending = sorted(release for release in releases if release > scalar)
        if len(pending) >= VIF_ENTRIES:
            scalar = pending[len(pending) - VIF_ENTRIES]
        releases = [release for release in releases if release > scalar]
        for item in items:
            if match := re.match(r's0: simm\.s32 (s\d+), (-?\d+)', item):
                registers[match.group(1)] = int(match.group(2))
            if match := re.match(r's0: srdreg\.lcclo s(\d+)', item):
                reads[int(match.group(1))] = scalar
        vector = [item for item in items if not item.startswith(('s0:', 's1:'))]
        fence = 's0: sfence' in items
        if not vector and not fence:
            scalar += 1
            continue
        # 含 TC VMEM 访问（vld*/vst*）的 bundle 最早在标量发射后 2 cycles 向量发射，其余 1 cycle；另须等待源操作数。
        memory = any(re.match(r'(?:vld|vst): v(?:ld|st)', item) for item in vector)
        issue = max(vector_free, scalar + 1 + memory)
        hold = 0
        for item in vector:
            sources = item.split(',', 1)[1] if ',' in item else ''
            issue = max([issue, *(ready.get(source, 0) for source in re.findall(r'\bv\d+\b', sources))])
            if match := re.match(r'misc: vdelay (\S+)', item):
                hold = registers[match.group(1)] if match.group(1).startswith('s') else int(match.group(1))
        for item in vector:
            if match := re.match(r'(?:va\d|vld|misc): (\S+) (v\d+),', item):
                ready[match.group(2)] = issue + RESULT_LATENCY.get(match.group(1), 1)
        vector_free = issue + 1 + hold
        releases.append(issue + RELEASE)
        # sfence 所在 bundle 在向量侧按序占一个位置；下一 bundle 等所有项（包括它自己）在标量侧释放。
        scalar = max(releases) if fence else scalar + 1
    return reads

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpu_latency_numbers/01_vld'))
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.output / 'results.jsonl').read_text().splitlines()]
    mismatches, samples = 0, 0
    for record in records:
        reads = replay(bundles(args.output / f"{record['name']}.tpuasm"))
        predicted = [reads[21] - reads[20], reads[22] - reads[20]]
        measured = set()
        for lcc, gaps, correct, words in zip(record['lcc'], record['gaps_from_begin'], record['correct'], record['matching_words'], strict=True):
            assert gaps == [lcc[1] - lcc[0], lcc[2] - lcc[0]], record['name']
            assert correct and words == 1024, record['name']
            measured.add(tuple(gaps))
            samples += 1
        if measured != {tuple(predicted)}:
            mismatches += 1
            print(f"{record['name']}: model R1-R0/R2-R0 = {predicted}, measured = {sorted(measured)}")
    print(f'{len(records) - mismatches}/{len(records)} configurations match; {samples} raw samples checked')
    assert mismatches == 0, '存在不符合逐周期发射模型的实测配置'

if __name__ == '__main__':
    main()
