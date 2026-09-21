# 本地 KEV：20 题 skill 选择测试包

整个文件夹可独立拷贝到执行机，不依赖原项目。只测试 20 条首次请求的“原始问题 → skill”，不发送计划和工具历史，不生成参数、不执行工具、不测试后续轮次。

## 直接运行

Python 3.10+。进入本文件夹执行：

```bash
python3 -m pip install -r requirements.txt
python3 run_benchmark.py run
```

已按你提供的示例配置默认参数：

```python
TypeSafeClient(
    api_key="local",
    base_url="http://71.77.153.223:55332",
    model="kev-latest",
)
```

每题调用 `client.system_one(state={"question": 原始问题}, questions={"skill": Choice(...)})`，从 `response.choices["skill"].choice` 读取选择。只发送 Choice，不发送 Noul/Score。候选描述来自原始日志，含四种业务工具及 __no_skill__ 兜底选项。

地址、模型与并发可修改：

```bash
python3 run_benchmark.py run \
  --base-url http://71.77.153.223:55332 \
  --model kev-latest \
  --workers 2 \
  --out output/kev_run1
```

`--base-url` 是 SDK 服务根地址，不要自行拼接 `/v1/systemone`。默认 key 是用户提供的本地占位值 local；如果需要其他 key，设置 `LOCAL_MODEL_API_KEY` 环境变量，或通过 `--api-key-env` 指定另一个环境变量名。包内没有官方 JEV 密钥。

## 离线准备与结果

```bash
# 只导出请求，不联网、不要求安装 SDK
python3 run_benchmark.py prepare

# 离线测试，使用模拟 SDK，不连接服务
python3 -m unittest test_benchmark.py -v

# 对默认输出目录重新统计，不请求模型
python3 run_benchmark.py report

# 自定义输出目录的统计
python3 run_benchmark.py report --out output/kev_run1
```

默认输出 `output/`：

- `manifest.json`：数据指纹、服务地址与模型，防止混用实验。
- `results.jsonl`：逐题预测、置信度、响应、耗时或错误类型。
- `summary.json`：一致率、成功/失败数、混淆矩阵、平均/p50/p95 耗时。
- `disagreements.jsonl`：与原模型不一致或失败的题目。
- `requests.jsonl`：prepare 时导出的请求（model 默认替换为 kev-latest）。

请求完成即保存；相同配置再次 run 跳过成功题目，重新请求失败题目。想完整重跑或更换模型、地址时，指定新的 `--out`。请求超时和单次调用内部重试使用 SDK 默认配置；脚本不叠加额外自动重试，耗时包含 SDK 调用期间的等待及客户端创建。

## 数据文件

- `questions.jsonl`：固定 20 题、候选 skill、原模型基线及原 JEV payload。
- `example_request.json`：默认本地模型对应的一条逻辑请求示例。
- `gold.template.jsonl`：可选人工标签模板。
- `run_benchmark.py` / `test_benchmark.py`：执行脚本与离线测试。
- `requirements.txt`：固定 `typesafe-sdk==0.7.0`。若执行机已安装并验证过其他 SDK 版本，可沿用现有环境；不兼容时再按此依赖安装。

为与此前官方 JEV 的 20 题测试可比，保留相同 question、候选描述和 instructions。instructions 仍有之前的历史判断措辞，但实际 state 始终只包含原始问题。原模型 baseline 只用于本地统计，绝不会加入模型请求。

SDK 调用依据：[官方 Python SDK 文档](https://docs.typesafe.ai/sdk/python)。本包只做过离线测试，尚未连接你的模型服务。

## 统计口径

与日志中的 `qwen3.6-27b-tp4-9.20` 首次 skill 选择比较，不重新调用原模型。`agreement_all` 的分母为全部 20 题，失败计为不一致；另提供仅成功请求的一致率。

没有人工标准答案，因此默认 accuracy 为 null。需要准确率时，将 gold.template.jsonl 复制为 gold.jsonl，人工填写 gold_skill 后执行：

```bash
python3 run_benchmark.py report --gold gold.jsonl
```

20 题基线全部为 query_param；其他候选 query_pm/query_alarm/calculate_gain 没有正例，因此不能用这一组的一致率证明整体多类分类准确率。此前官方 JEV 相同请求的参考结果为 20/20 一致、平均耗时 1.400 秒。历史原模型耗时包含参数生成，不能直接作为等价工作量的速度对照。
