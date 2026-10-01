"""从串行 BC 程序生成 W 个独立目的切片的 issue-all / wait-all 批次。"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

def window_program(space: str, size: int, window: int) -> str:
    base = HERE / ('bmem_loop.bca' if space == 'hbm' else 'bc_pull_loop.bca')
    text = base.read_text()
    if window == 1:
        return text
    text = text.replace('registers=5-6,10-15,20-25', 'registers=5-6,10-15,20-29')
    text = text.replace('%s14 = smov 9\n', f'%s14 = smov 9\n%s28 = smov {size * window // 512}\n')
    text = text.replace('%s14 = smov 11\n', f'%s14 = smov 11\n%s28 = smov {size * window // 512}\n')
    def block(name: str, lines: list[str]) -> str:
        return f'.scalar_block {name}\n' + '\n'.join(lines) + '\n.endscalar_block\n'

    def bundle(instruction: str, slot: str = 's0') -> str:
        return f'.bundle\n.{slot} {instruction}\n.endbundle\n'
    gap = '.bundle\n.endbundle\n.repeat 4\n'
    hot = ''
    for w in range(window):
        hot += block(f'reset_window_{w}', [f'%s13 = smov {16+w}', 'sst [%s13] %s12'])
        hot += gap + bundle('sdone.write %s13, %s12', 's1')
    for w in range(window):
        if space == 'hbm':
            hot += block(f'address_window_{w}', [f'%s13 = smov {16+w}', f'%s29 = smov {w * size // 512}']) + gap
            hot += bundle('scalar_dma_simple c1:v=15 c2:v=1 c4:v=0 a1:v=10 a2:v=15 a3:v=29 a4:v=13 a5:v=0 a6:v=0 a7:v=0 a8:v=0 a9:v=0 a10:v=1 a11:v=0 a12:v=0 a13:v=0 a14:v=0 a15:v=0 a16:v=0 a17:v=0')
        else:
            start = text.index('.staged_descriptor')
            descriptor = text[start:text.index('.enddescriptor', start) + len('.enddescriptor')] + '\n'
            descriptor = descriptor.replace('destination_sync_flag_0_id=10', f'destination_sync_flag_0_id={16+w}')
            descriptor = descriptor.replace('destination_address=0', f'destination_address={w * size // 512}')
            hot += descriptor
        hot += block(f'advance_window_{w}', ['%s10 = sadd.s32 %s10, %s15'])
    for w in range(window):
        hot += block(f'wait_window_{w}', [f'%s13 = smov {16+w}']) + gap + bundle('swait.done %s13')
    hot += bundle('scalar_fence c1:v=15 c2:v=1 c4:v=0 a1:v=32 a2:v=255 a3:v=0 a4:v=0 a5:v=0')
    a = text.index('hot_loop:\n') + len('hot_loop:\n')
    b = text.index('.scalar_block advance_count', a)
    text = text[:a] + hot + text[b:]
    if space == 'hbm':
        text = text.replace('%s10 = sadd.s32 %s10, %s24\n', '')
    else:
        a, b = text.index('.scalar_block advance_source'), text.index('validate:\n')
        text = text[:a] + block('next_batch', ['%s10 = smov %s25', 'sbr.rel @hot_loop']) + text[b:]
        # Release still uses flag 10 and the full producer allocation.
        text = text.replace('.scalar_block reset_release\n', '.scalar_block reset_release\n%s13 = smov 10\n')
        release = text.index('.scalar_block reset_release')
        text = text[:release] + text[release:].replace('.field size=sreg(15)', '.field size=sreg(28)')
    text = text.replace('a1:v=12 a2:v=15 a3:v=11', 'a1:v=12 a2:v=28 a3:v=11')
    return text.replace('.pad_to 128', '.pad_to 2048').replace('.pad_to 256', '.pad_to 2048')
