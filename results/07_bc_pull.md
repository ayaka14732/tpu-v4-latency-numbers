TC VMEM（`vmem`，芯片 (0,0) 上运行中的 Pallas kernel 持有的 scratch）或 Megacore Shared CMEM（`cmem`）→BC 私有 BMEM，由 BC 发起，libtpu 0.0.50，tpuasm 的 `BarnaCoreRuntime`；同一 BC 的 paired LCC 包围 W 条 `dma.simple` 及其完成 sflag 的 `swait.ge`；TC VMEM 来源在 BC 读完后由 BC 归还 kernel 的信号量。S 为每条字节数，K=S/1024；W=1/4/16 的 S 分别为 4 KiB–2 MiB、4–512 KiB、4–128 KiB。以下为冻结中心公式及独立尺寸残差，共 1392 个样本。[复现与完整条件](../README.md#7-tc-vmemmegacore-shared-cmem--bc-私有-bmem)。DMA 尺寸须为 4 KiB 的整数倍；公式针对正文列出的已测条件。libtpu 0.0.46 的旧公式见[历史对照](07_bc_pull_libtpu_0_0_46.md)。

| 来源 | W | raw 周期中心公式，K=S/1024 | 验证中位数最大误差 | 验证全样本残差 |
| --- | ---: | --- | ---: | --- |
| cmem | 1 | 255.175 + 13.33097K | 6.18 | [-4.95,8.18] |
| cmem | 4 | 251.194 + 53.33402K | 1.46 | [-1.46,2.13] |
| cmem | 16 | 251.367 + 213.33270K | 0.31 | [-1.31,1.97] |
| vmem | 1 | 310.564 + 13.33319K | 0.76 | [-1.55,1.77] |
| vmem | 4 | 310.341 + 53.33362K | 0.63 | [-1.45,1.99] |
| vmem | 16 | 310.011 + 213.33470K | 1.14 | [-1.14,2.31] |
