HBM→host，host 发起，libtpu 0.0.50：`api` 调用 tpuasm 的 `BarnaCoreRuntime.read_hbm`（含它分配缓冲区并复制成 `bytes` 的 Python 开销），`native` 把预先分配的缓冲区直接交给同一个原生桥接调用（`TfrtDriverWrapper::ReadHbm`，即同步的 `TpuSharedMemory::Read`）。同一 BC 上的常驻计数程序以 paired LCC 包围两次 host↔BC 握手与其间的一次读取，数据在 END 之后核对。S 为 4 KiB–4 MiB，源为 64 MiB 的 HBM 环。只握手、不读取的空窗口 24 个样本中位数 54990、范围 [54520, 59518]，不从公式中扣除。这不是第 9 节的 Host Magic Queue 路径，[说明](../README.md#92-libtpu-0050-上的-host-读取)。

| 调用 | W | raw 周期中心公式，K=S/1024 | 验证中位数最大误差 | 最大相对误差 | 验证全样本残差 |
| --- | ---: | --- | ---: | ---: | --- |
| api | 1 | 103623.120 + 335.16983K | 57006.93 | 7.22% | [-65382.93,116833.07] |
| native | 1 | 110646.539 + 237.99153K | 24628.80 | 5.77% | [-1194.27,170185.73] |
