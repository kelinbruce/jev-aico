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
