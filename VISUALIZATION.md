# 周期线图与公式表

页面标题为「TPU v4 数据搬运周期」。横轴为数据量，纵轴为周期数，多条路径或条件画在同一坐标系中。页面只保留标题、必要筛选器、图例和公式表；测量条件默认折叠，没有眉题、副标题或结论摘要。

## 图与数据

| 图 | 横轴 | 同图比较 | 筛选条件 | README 章节 |
| --- | --- | --- | --- | --- |
| 本地 DMA | 每条 DMA 字节数 S | 六条固定地址路径；或四条 HBM 环形工作集路径 | 固定地址／每 TC 64 MiB HBM 环；W=1/4/8；单 TC／双 TC 中指定的一核 | 2、10 |
| 片内与跨芯片 DMA | 每条 DMA 字节数 S | 内存来源及各发起流 | 全部 23 种拓扑；W=1/4/8 | 4 |
| BC 私有 BMEM | 每条 DMA 字节数 S | HBM、TC VMEM、Megacore Shared CMEM 三种来源 | W=1/4/16 | 5、7 |
| HBM → Host | 每条 DMA 字节数 S | 同一协议的 W=1/4/16；Magic Queue 另含 W=64 | TC 发起／Host Magic Queue；初始测量／分段模型复测 | 6、9 |
| 完整 payload 往返 | 单程字节数 S | 六个跨芯片 pair、四个片内 pair、三种并行双 pair 的各发起端 | 拓扑类别 | 8 |
| 指令流读取 | 逻辑读取字节数 B=4096N | 普通 vld 与串行 cld 累加；不同预取深度的 cld | 独立／合并 consumer；常用／深预取 | 1、3 |
| 循环读取 | 逻辑读取字节数 B=4096UN | TC VMEM 与 Megacore Shared CMEM | 64/256/1024/4096 KiB 工作集；单／双 TC；全部已测 U/A 组合 | 11 |

片内与跨芯片图包含全部 12 个有向芯片对、四颗芯片的片内单向及双向、独立双流、全双工和同一芯片对双 TC 流；只有实测了两种内存的拓扑才同时显示两条来源。循环读取保留完整 52 个配置，含 A=8 展开对照和仅有 Megacore Shared CMEM 的 A=2 对照。筛选器不提供未测的工作集／U/A 组合。

横轴使用真实数值比例，默认线性，可切换对数。每个点为同一尺寸同一配置的实测中位数，按数据量排序后连线；DMA 和循环读取每点 24 次，早期 CLD 探针每点 12 次，其余指令流探针 24 次。悬停显示数据量、周期、采样用途及重复次数。只连接各条线自己的实测范围，不外推、不生成新采样、不把小数公式当作实测数据。第 0 节的计数器校准、端点负对照和 load-use 等机制细节仍在 README 查表，不另设诊断图。

周期均为本地 LCC 区间，不相加不同核心的读数。窗口 W 表示一批有 W 条 DMA，每条大小 S；完整 payload 往返共搬运 2S。Host 的 TC 计数区间与 Magic Queue 的 BC 计数区间通过发起方式切换，同图只比较同一协议的窗口。分段模型复测显示新增采样和对应冻结公式，W=1 沿用初始数据。BMEM 三种来源的最大已测大小不同，范围分别列出。

指令流图的 vld 不含累加，cld 含 vpop 和整数累加，表中给出各自完整指令流的成本；深预取包含装填和排空。循环读取包含 float32 累加、寻址、循环和 fence；其横轴不是工作集大小，公式同步写成 C(B)，保留 P=BEGIN mod 2 和 U=A=32 时显式 spill 的条件。

各图下直接列公式、大小范围及必要的独立验证残差。另将仓库中的 12 份正式分表解析为 13 张可筛选表：409 条逐条件中心公式，加 4 条 Host Magic Queue 空窗口记录。空窗口表使用自己的三列表头，合计 413 条有效记录。寄存器图采用逐样本通过的整数公式，原有逐配置中心公式仍单独保留。

## 一图总结

[`latency_numbers.svg`](latency_numbers.svg) 是对外介绍用的单张英文图，标题为「Latency Numbers Every TPU Programmer Should Know」。它不替代本页的交互图，只是总览，列出 17 条最常用路径：TC VMEM 与 Megacore Shared CMEM 读入 TC VREG（后者取 README 第 3.4 节 D=27 深流水的 `2N+66`，即 `66 + K/2`，图中注明 `K ≥ 108`；4 KiB 单个向量无法流水，按 67 cycles 标注）、七条本芯片单条 DMA、TC VMEM 的 1 跳与 2 跳 ICI 和 Megacore Shared CMEM 的 1 跳 ICI、由 BC 发起的 HBM／TC VMEM／Megacore Shared CMEM → BC 私有 BMEM，以及 TC 发起的 HBM → pinned Host 和 Host Magic Queue 发起的 HBM → 预映射 Host buffer。图中为节省篇幅，把 Megacore Shared CMEM 简写为 CMEM，BC 私有 BMEM 简写为 BC BMEM。每行给出 `a + b·K` 形式的周期公式：绿色 `=` 表示逐周期成立的整数公式（两条读入 TC VREG 的指令流和 Megacore Shared CMEM ↔ TC VMEM 两条 DMA），灰色 `≈` 表示先冻结、再经独立尺寸验证的中心公式，图例分别写作 exact 与 fitted；横向对数轴标出 4 KiB 与 1 MiB 时的位置，并在两端注明周期数；BC 计时的各行（第 5、7、9 节）已按 README 第 12 节测得的同芯片 BC/TC 计数比例换算为 TC 周期。图末一行说明公式形式、K 的单位、cycle 与 ns 的换算、两种点和两种关系符的含义。公式、刻度、周期数和图末的公式说明行按 TeX 的排版方式绘制：数字直立、变量为数学斜体，运算符与关系符两侧留 TeX 的标准间距，字形取自 TeX 默认数学字体的 OpenType 版 Latin Modern Math，并作为轮廓嵌入 SVG，查看者无需安装字体。各路径的条件和残差只在 README 与本页交互图中说明。

```sh
/srv/workspace/venv/bin/python scripts/14_summary.py
```

脚本只读取仓库内 `results/` 的分表，不依赖 `/tmp` 中的采样；它需要 `fontTools`，首次运行时用 `curl` 从 CTAN 下载 Latin Modern Math 到 `/tmp/tpu_latency_numbers/fonts/`，字体不提交到 Git。寄存器流与 Megacore Shared CMEM ↔ TC VMEM 的整数公式取自 README 第 1、2、3 节。SVG 内含浅色与深色两套配色，随查看者的系统主题切换；需要 PNG 时在浏览器中按 1200 px 宽截图即可，PNG 不提交到 Git。

## 生成和预览

```sh
/srv/workspace/venv/bin/python scripts/13_visualize.py
/srv/workspace/venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory /tmp/tpu_latency_numbers/visualization
```

浏览器访问 `http://localhost:8765`；远程 VS Code 先转发 8765 端口。已有预览服务时只需重新生成并刷新网页。也可下载生成的 `index.html` 后离线打开。页面内嵌已有 Chart.js 和聚合数据，无 CDN 或后端依赖；`#bmem`、`#host`、`#rtt` 等锚点可直接定位对应图。

[生成脚本](scripts/13_visualize.py)只读取已有采样和冻结模型，在 CPU 上核对原始计数差、完整数值通过记录、经验模型残差和寄存器整数公式，不启动 TPU、不拟合。`--root` 指定采样根目录，`--output` 指定 HTML 路径。新 checkout 需先按 README 第 1–11 节复现对应采样；正式分表直接从仓库读取。

Git 保存代码、模板、正文和精简分表；原始 JSON、完整汇编、数组、生成页面、检查记录与截图留在 `/tmp` 或指定的仓库外目录。

浏览器检查逐项切换全部图表条件，核对实测点、数据量换算、模型对应关系和公式筛选，包括 54 条固定地址本地模型、36 条 HBM 环形模型、138 条片内／跨芯片模型、9 条 BMEM 模型、12 条 Host 模型、16 条 RTT 模型、20 组指令流和 52 个循环读取配置（144 条来源／TC 公式）。1440 px 桌面和 390 px 窄屏截图已检查，默认无可见副标题段落、无横向页面溢出，公式宽表在表内滚动。检查记录与截图位于 `/tmp/tpu_latency_numbers/visualization/`。
