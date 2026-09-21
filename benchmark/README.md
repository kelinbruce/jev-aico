# 只选择 skill：JEV benchmark

目标：从日志提取原始用户问题，让 JEV 选择 skill，与日志中原模型首次选择的 skill 对比。默认只处理首次请求，不发送执行计划或工具历史，不生成参数、不执行工具、不处理后续轮次。

Python 3.9+，仅使用标准库。在 `/Users/zhangfan/project/jev` 运行：

```bash
# 提取问题并转换请求，不联网
python3 benchmark/jev_skill_benchmark.py prepare

# 请求 JEV 并比较 skill（成功结果会复用，失败请求重试）
python3 benchmark/jev_skill_benchmark.py run --workers 2

# 只统计已有结果
python3 benchmark/jev_skill_benchmark.py report
```

默认读取 `data/4_0921/qwen3.6_modify_system_prompt_no_lora_delet_router.jsonl`，输出至 `benchmark/output/question/`。本文件共 41 条日志，但仅 20 条是首次请求，其他 21 条后续请求不参与此 benchmark。

## 转换规则

1. 从首次请求的 `request_body.messages` 的 `<task>...</task>` 提取原始问题，解码转义。
2. 从 `request_body.tools[].function` 提取候选 skill 名称和描述。
3. JEV 的 `state` 仅包含 `question`；`questions.skill` 使用 Choice 类型，候选为日志中的四种 skill，加上 `__no_skill__`（无适用工具的兜底选项）。
4. 从返回的 `answers.skill.choice` 读取 skill。
5. 与日志 `response_body.choices[0].message.tool_calls` 的工具名比较。原模型的本轮答案不会发送给 JEV。

协议参考：[TypeSafe 官方 API 文档](https://docs.typesafe.ai/api)。本次实测结果见 [RESULTS.md](RESULTS.md)。

## 运行配置与结果

key 按 `JEV_API_KEY`、`TYPESAFE_API_KEY`、项目根目录 `jev_api_key.txt` 的顺序读取。`run` 将原始问题及候选描述发送到官方 `https://api.typesafe.ai/v1/systemone`，可能产生 API 费用。

- `--input`：指定输入文件。
- `--out`：指定独立输出目录。更换样本或模型时请使用新目录，避免与旧结果混用。
- `--model`：默认 `jev-latest`，可指定固定版本。
- `--limit 5 --seed 42`：可复现地抽样 5 条；建议同时指定独立 `--out`。
- `--workers` / `--timeout` / `--retries`：并发数、单次超时、重试次数。

产物包括 `dataset.jsonl`（问题、基线、请求 payload）、`results.jsonl`（真实响应、预测和耗时）、`summary.json`（汇总）、`disagreements.jsonl`（差异样本）。即时保存每次结果，重复 run 跳过成功请求。

## 准确率口径

无人工标签时统计与原模型的一致率，不能将原模型输出等同于正确答案。
将输出目录中的 `gold.template.jsonl` 复制为 `gold.jsonl`，按原始问题人工填写 `gold_skill` 后运行：

```bash
python3 benchmark/jev_skill_benchmark.py report --gold benchmark/output/question/gold.jsonl
```

程序会对同一组标注样本计算 `jev_accuracy` 和 `baseline_accuracy`；未标注为 null，失败计错。当前 20 条样本全部属于 query_param，需补充其他 skill 的样本才能衡量多类分类能力。JEV 只选 skill，而原模型的历史耗时包含参数生成，因此耗时不构成同等工作量的严格对照。

验证命令：

```bash
python3 -m unittest discover -s benchmark -p 'test_*.py' -v
```

兼容保留 compact/full 和 phase 参数以读取早期实验，不属于当前任务的默认流程；早期 `output/compact/` 结果仅作存档。
