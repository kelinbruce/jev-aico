# JEV AICO

AICO skill 选择数据、JEV 对比 benchmark 和本地 KEV 测试工具。

## 本地模型：20 题 skill 选择

进入 `skill_selection_20/`，Python 3.10+：

```bash
python3 -m pip install -r requirements.txt
python3 run_benchmark.py run
```

默认使用已配置的本地服务、`kev-latest` 和本地占位 key。详见 [测试包说明](skill_selection_20/README.md)。不发送执行计划或工具历史，只评估首次 skill 选择。

## 目录

- `skill_selection_20/`：可独立运行的 20 题本地 SDK 测试包。
- `benchmark/`：原始日志转换、官方 JEV benchmark 与实测结果。
- `data/`：原始调用日志、分析脚本与报表。
- `demo/`：JEV Choice / Score / Noul 示例。
- `nimble/`：来自 https://github.com/bespokelabsai/nimble 的源码快照，保留其许可证及项目文档。

官方 API key 不入库；使用官方 JEV 示例时自行配置。数据包含业务问题及历史响应，本仓库使用私有可见性。
