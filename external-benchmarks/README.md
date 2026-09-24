# 外部 Jev 实现对照测试集

10 个固定测试集，共 **2,308 条记录 / 2,308 个问题**。下载仓库即可运行，不需要重新下载 Hugging Face 数据。

优先使用常见公开任务的固定子集，与 Bespoke Nimble 和 Jev 的已发表评测对齐。七个公开子集均按 Nimble 发布的 ID 清单还原，**转换前的数据 SHA-256 与原始 manifest 完全一致**；仅换成现有 TypeSafe 测试脚本的存储格式，保留 state、instructions、criteria、候选项顺序和标准答案。

## 包含什么

下表外部成绩是作者报告，不是本仓库重新测试的结果。准确率以问题为单位。

| 测试集 | 条数 | 能力 | Nimble-9B | Jev 1.13.0 |
|---|---:|---|---:|---:|
| boolq | 300 | 从文章判断 yes/no | 86.0% | 89.7% |
| paws | 250 | 高词汇重叠的句子是否同义 | 82.8% | 89.2% |
| squad2 | 299 | RAG：上下文是否足以回答 | 80.6% | 82.9% |
| pubmedqa | 250 | 医学摘要 yes/no/maybe | 75.6% | 77.2% |
| helpsteer2 | 249 | 回答帮助程度，五级评分 | 39.0% | 34.1% |
| summeval-relevance | 240 | 摘要相关性，五级评分 | 49.2% | 35.0% |
| summeval-consistency | 144 | 摘要事实一致性，五级评分 | 75.7% | 81.2% |
| nimble-synthetic-324 | 324 | 合成的规则与证据决策 | 90.12% | 93.21% |
| semif-authored-144 | 144 | SemIf 构造的语义决策 | — | 96.53%¹ |
| semif-perturbations-108 | 108 | 选项重排、指令包裹、无关上下文 | — | — |

七个公开子集来源：[Nimble Public Benchmarks](https://github.com/bespokelabsai/nimble/blob/main/docs/PUBLIC_BENCHMARKS.md)。报告快照保存在 `sources/NIMBLE_PUBLIC_BENCHMARKS.md`，每个数据集下均有原始抽样清单和校验值。

合成集来源：[Nimble 324 条留出集](https://github.com/bespokelabsai/nimble#evaluation-on-324-held-out-examples)。标签是合成参考标签，并非人工审核。

¹ SemIf 的 Jev 96.53% 来自 [Kev 保存的外部评测](https://github.com/jaredpalmer/kev/blob/main/runs/jev-semif-v1/report.json)，是 139/144 的普通准确率，该报告未在此固定 Jev 的精确服务版本。**SemIf 自身发布的 Qwen3.5-4B 81.3%、Qwen3-Reranker-4B 62.5% 是 mean family balanced accuracy**，需要看本脚本的 `family_balanced_accuracy`，不能拿普通 accuracy 直接比较。[SemIf 原始结果](https://github.com/TheoLeeCJ/SemIf/blob/master/docs/RESULTS.md)。108 条扰动样本是 36 个原始案例各做三种变换，不把它与 36 条基准案例的分数混用。

## 运行

在仓库根目录，Python 3.10+：

```bash
python3 -m pip install -r external-benchmarks/requirements.txt

# 列出所有集合
python3 external-benchmarks/run_benchmarks.py --list

# 先对每个集合各跑 3 条，确认服务可用
python3 external-benchmarks/run_benchmarks.py --limit 3

# 全部 2,308 条
python3 external-benchmarks/run_benchmarks.py

# 只跑几个常见任务
python3 external-benchmarks/run_benchmarks.py --suite boolq paws squad2

# 换成其他兼容 TypeSafe 的模型服务
python3 external-benchmarks/run_benchmarks.py --base-url http://127.0.0.1:8009 --model nimble

# 只检查数据校验值、SDK 请求格式，不调用模型
python3 external-benchmarks/run_benchmarks.py --dry-run
```

默认地址 `http://127.0.0.1:55733`，模型 `kev-latest`，API key 为 `local`；认证通过环境变量 `LOCAL_MODEL_API_KEY` 设置。每次请求超时 300 秒，重试 0 次，默认串行。SDK 会调用 `/v1/systemone`，base URL 不要重复带上这个路径。SemIf 等没有此 API 的原生实现需要额外的接口适配，不能只改模型名。

入口复用 `decision-v7/run_test.py`，请保留这两个目录。结果按时间新建目录，不覆盖旧结果；发生请求错误会保留已完成输出，并停止后续集合。

## 在哪里看结果

```text
external-benchmarks/output/<时间戳>/
├── REPORT.md               # 本地与外部成绩并排，标注指标、差值
├── comparison.json         # 汇总、外部报告链接与是否全量可比
└── <测试集>/
    ├── summary.json        # accuracy、score_mae、错误数、耗时
    ├── results.jsonl       # 标准答案、预测、原始响应、逐题错误
    └── run.json            # 模型、地址、数据 SHA-256
```

`accuracy` 为 0–1。`score_mae` 是概率加权的等级期望值与标准等级的平均绝对差，越低越好，不是把 `score` 四舍五入后算准确率。外部表格已舍入，极小差值不代表显著差异。

有限条数试跑、未完成或含错误时不计算对外部基线的差值。全量运行也只是同样本、同题面、同指标的参考比较；模型版本、精度、概率舍入和服务预处理仍可能影响结果。多数集合是英文；不据此推断中文能力。合成集与公开人工标注集分别看，不提供混合总排行榜。

Noul 使用 p(true)>0.5，恰好 0.5 时判 false；Score 取最大概率等级，并列取较低等级；Choice 使用服务返回的 choice。只发送 state 和题目字段，标准答案、元数据、教师输出均不会发送。请求失败计入 accuracy 分母，但不进入 score_mae；请同时查看错误数。

## 数据来源与许可

- BoolQ：Google / BoolQ，CC BY-SA 3.0。[来源](https://huggingface.co/datasets/google/boolq)
- PAWS：Google Research，原项目声明允许任意用途。[来源及许可](https://github.com/google-research-datasets/paws)
- SQuAD2：Stanford / SQuAD 2.0，CC BY-SA 4.0。[来源](https://huggingface.co/datasets/rajpurkar/squad_v2)
- PubMedQA：PubMedQA 作者，MIT。[来源](https://huggingface.co/datasets/qiaojin/PubMedQA)
- HelpSteer2：NVIDIA，CC BY 4.0。[来源](https://huggingface.co/datasets/nvidia/HelpSteer2)
- SummEval：SummEval 作者，经 MTEB 发布，MIT。[来源](https://huggingface.co/datasets/mteb/summeval)
- SemIf：TheoLeeCJ，MIT；本次使用 Kev 的 Apache-2.0 转换快照，原始提交和校验值见 manifest，附带 `sources/KEV_LICENSE`。
- Nimble 合成集：Bespoke Labs；本地快照未找到独立的数据许可证，保留来源说明，不将其重新声明为 MIT/Apache；进一步分发前核对上游条款。

格式转换不改变数据内容含义，原数据仍受各自许可约束。CC BY-SA 数据的格式转换版本沿用原对应许可。生成器基于仓库中的 Nimble 原始转换模块；固定子集保留作者题面和候选项。

## 复现公开子集

运行测试无需这些步骤。只有重新从上游构建时才需要 PyArrow 和仓库中的 `nimble/` 源码：

```bash
python3 -m pip install pyarrow
python3 external-benchmarks/fetch_public.py --raw external-benchmarks/raw
python3 external-benchmarks/prepare_public.py --raw external-benchmarks/raw
```

生成器按原始 ID 选取，比较完整转换前子集的 SHA-256；如果上游内容或转换模块变化导致不一致会失败，不能悄悄换成另一批题。下载来源和 Parquet 校验值记录在各集合 manifest 内。大体积上游原始数据不入库。
