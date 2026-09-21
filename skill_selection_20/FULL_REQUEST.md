# 从真实 request_body 进行完整上下文测试

`run_full_request.py` 使用 Python 3.9+ 标准库，不需要 SDK。默认读取仓库中的 `data/4_0921/qwen3.6_modify_system_prompt_no_lora_delet_router.jsonl`，默认自建服务为 `http://71.77.153.223:55332`，模型为 `kev-latest`。

默认仅选择尚无 assistant/tool/function 历史的首次请求；当前日志 41 条中符合条件的 20 条与原 20 题测试集完全一致，排除 21 条后续调用。`--limit` 在筛选之后生效。

从每条记录的 `request_body` 仅提取 `messages` 和 `tools`，放到 KEV 请求的 `state.request_body`。完整保留全部 system/user/assistant/tool 消息和工具参数 schema。存在 system 消息时保留；原始记录只有 user 消息时不会虚构 system 消息。`questions.skill` 从真实工具名称和描述构建候选，并补充无需调用工具选项。日志中的本轮 `response_body` 不发送。

原始 `verify_ssl`、`timeout`、`model`、`temperature`、`max_tokens`、`top_p`、`tool_choice`、`chat_template_kwargs`、`enable_thinking` 等其他字段均不拼接。外层 KEV `model` 仍使用 `--model`；HTTP 超时由脚本 `--timeout` 控制。这是完整上下文的 skill 选择测试，仍只返回工具名称，不生成参数或执行工具；不是原始 Chat Completions 协议的原封重放。

在本文件所在目录运行：

```bash
# 导出第一条完整请求，不联网
python3 run_full_request.py prepare --limit 1 --out output_full_preview

# 单条实际调用，使用另一个输出目录
python3 run_full_request.py run --limit 1 --out output_full_run1

# 全部 20 道首次 skill 选择题，单并发调用
python3 run_full_request.py run --out output_full_20

# 自定义真实日志文件、起始行与条数
python3 run_full_request.py run --input /path/to/real.jsonl --start-line 2 --limit 1 --out output_full_custom

python3 -m unittest test_full_request.py -v
```

固定单并发，无自动重试；每次请求默认超时 60 秒，可用 `--timeout` 修改。key 从 `LOCAL_MODEL_API_KEY` 读取，未设置则使用 `local`；可用 `--api-key-env` 指定变量。`--base-url` 指定服务根地址。每次执行使用新的输出目录，不覆盖已有批次。

输出 `requests.jsonl` 每行就是完整实际请求体，可直接用于 HTTP POST `/v1/systemone`；`sources.jsonl` 按相同行号记录来源行、原请求 ID、请求 SHA256 和字节数；`manifest.json` 记录运行配置；`results.jsonl` 逐条保存响应或 HTTP 状态码、错误详情、请求 ID、耗时。错误详情最多 8192 字节，并遮盖当前 API key。

## Benchmark 统计

`run` 完成后自动在终端打印统计，并保存 `summary.json` 和 `disagreements.jsonl`。

对已经执行的结果补统计，不联网、不重复请求模型：

```bash
python3 run_full_request.py report --out output_full_20
```

旧版本结果同样支持；旧批次需保留原始日志，若执行时指定了自定义输入，report 也传入相同的 `--input`。统计按输出目录保存的请求清单进行，不受本次 `--limit` 影响。中断批次未完成的题目计入 missing。

汇总包含总数、成功/失败/缺失数、成功率、与原模型选择一致率、混淆矩阵、成功请求及全部已执行请求的平均/p50/p95耗时（毫秒）。`agreement_all` 以有原模型参考答案的全部题目为分母，失败和缺失算不一致；`agreement_success_only` 仅统计有参考答案且调用成功的题目。原模型参考答案仅用于本地统计，不加入请求体。

一致率不等于准确率，未提供人工标签时 `accuracy` 为 null。当前 20 题原模型参考答案均为 query_param，不能用此一致率代表整体多类 skill 选择能力。
