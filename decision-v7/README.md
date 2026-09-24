# decision-v7 基础准确率回归集

来自 Kev 的 `evals/v7/decision-v7/development.jsonl`，用于意图分类、新闻分类、情感分析、文本推断和规则决策等基础准确率回归。

- 记录数：1,204。
- 问题数：1,468（单条记录可包含多个问题）。
- 分区：development，保留原始文件名和字节内容。原套件的 test 分区有 1,176 条，不包含在此目录中。
- 分布：来源与 Kev 训练分布有关，适合作为同分布基础回归集，不应据此推断跨分布泛化能力。

## 文件

- `development.jsonl`：本次上传的完整 1,204 条数据，每行一个 JSON 对象，包含 `state`、`questions`、`_meta`。
- `manifest.json`：原套件清单，保留数据源版本、分区数量和 SHA-256；其中也描述了本目录未收录的 train、calibration、test 分区。
- `LICENSE`：Kev 项目附带的 Apache-2.0 许可证；各上游数据集的来源与版本见 manifest 和记录中的 `_meta`，使用时应遵循各自许可。

## 完整性

`development.jsonl` 的 SHA-256：

```text
8d5765d7aec4d08c61854f4664ca79ec2ba44ed092e9b967eaf61ed86c496d9c
```

本目录仅复制原始数据，未改写问题、标签或元数据。

## 启动本地测试

需要 Python 3.10+，先启动本地 Kev 服务，然后在仓库根目录执行：

```bash
python3 -m pip install -r decision-v7/requirements.txt
python3 decision-v7/run_test.py
```

默认连接 `http://127.0.0.1:55733`，模型为 `kev-latest`，API key 为 `local`，超时 300 秒，不重试。若启用认证，用环境变量 `LOCAL_MODEL_API_KEY` 提供 key。

```bash
# 先测试前 5 条
python3 decision-v7/run_test.py --limit 5
# 仅校验数据与 SDK 请求格式，不发送请求
python3 decision-v7/run_test.py --dry-run
# 覆盖服务地址
python3 decision-v7/run_test.py --base-url http://127.0.0.1:55733 --model kev-latest
```

每条记录发送一次请求，保留同一记录内的多题结构；`label`、`src` 和 `_meta` 不发送给模型。默认串行执行。

结果保存在 `decision-v7/output/<时间戳>/`：`run.json` 记录运行配置，`results.jsonl` 逐条保存预测、标准答案、错误、原始响应及请求耗时，`summary.json` 汇总总准确率、按来源/题型准确率和延迟。可使用 `--out 新目录` 指定输出位置，已有目录不会被覆盖。Ctrl+C 会保存已完成记录的汇总。

准确率按问题计算：Choice 比较选择标签；Noul 概率大于 0.5 为真（等于 0.5 时按 false）；Score 取概率最高的等级（并列取较低等级），不对期望分数四舍五入。服务返回的概率可能经过取整，临界并列时可能与模型内部完整精度评分不同。失败题在总准确率中按错误计，另列成功题准确率。首个请求失败时提前结束，避免服务不可用时继续等待整批测试。
