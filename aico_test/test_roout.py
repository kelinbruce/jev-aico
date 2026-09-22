from typesafe_sdk import Choice, TypeSafeClient
import pandas as pd


client = TypeSafeClient(
    api_key="local",
    base_url="http://71.77.153.223:55333",
    model="kev-latest",
)

file_name = r"D:\case-2026techProgram\case-930LUI\测评\测试150.xlsx"
output_file = r"D:\case-2026techProgram\case-930LUI\测评\测试150_分类结果.xlsx"

state = """
基础查数：查询网络对象（如小区、站点等）的对象清单、运行性能指标、工参和状态数据。包括：1) 对象清单及名称、标识、ID、数目——按频段、覆盖类型（室内/室外）、设备型号等条件筛选枚举实际小区/站点；2) 工参（EP）设置和数值型运行数据——KPI 指标、PRB 利用率、吞吐率、流量、话务量、RRC、切换、掉话、接通率、VoLTE、干扰、覆盖、能耗、告警记录等原始数据，支持基于数据或类型的排序、过滤、TOP-N、按时间段/粒度统计（如过去12小时每小时）等简单聚合。
智能问答：【功能范围】支持回答期望了解华为无线领域相关的知识信息，包括原理、概念、功能说明、操作方法、流程步骤或使用指导的问题。【判别依据】命中依据：1、当咨询理论基础知识，而非需要借助工具查询实时数据时，应命中本意图。2、若问题提问华为无线、云核心网领域相关的知识信息，包括原理、概念、功能说明、操作方法、流程步骤或使用指导等，问题重点在询问解释性内容，而非直接获取系统中的实时或历史数据结果，非借助工具操作动作，应命中本意图。3、在无法匹配其他更具体意图的情况下，仅包含术语/告警/KPI名词、无明确动作动词的输入，默认命中本意图做概念解释。禁止命中依据：1、本意图不支持直接获取系统中的实时或历史数据结果。2、本意图不支持查告警源，告警源属于需要实时查询的信息。
拓扑与维度查询：仅支持按区域、制式两个维度筛选的站点名、小区名及其数目：1. 查xx区域的站点名、小区名；2. 查xx制式的站点名、小区名；3. 查xx区域、xx制式的站点名、小区名及其数目。以及指标名称列表：4. 查xx制式、xx区域的PM(性能)和EP(工参)指标名列表（不包含数据）；5. 查看系统支持的区域名、站点名、小区名、指标名。**制式参数仅指 2G/3G/4G/5G 及其对应的英文别名（如 2G-GSM、3G-WCDMA/CDMA2000/TD-SCDMA、4G-LTE、5G-NR），与频段（如 900M、1800M、2.6GHz）是两个不同的筛选参数——按频段筛选不属于制式筛选。**不包含数值数据；查询站点/小区时只要含区域、制式以外的筛选条件（含按频段筛选），或名称列表以外的数据，即应属于“指标与运行数据查询”意图。
其他：其他所有类型的都为其他
"""


df = pd.read_excel(file_name)

if "问题" not in df.columns:
    raise ValueError("Excel 中不存在 问题 列")

results = []

for index, row in df.iterrows():
    query = str(row["问题"]).strip()

    # 跳过空值
    if not query or query.lower() == "nan":
        results.append(None)
        continue

    instructions = f"""
请根据上面给出的业务类型定义，判断下面的用户问题属于哪一种类型。

用户问题：
{query}

只能从 criteria 给出的两个个选项中选择一个最符合的类型。
""".strip()


    response = client.system_one(
        state=state,
        questions={
            "intent": Choice(
                instructions=instructions,
                criteria={
                    "option_a": "基础查数",
                    "option_b": "智能问答",
                    "option_c": "拓扑与维度查询",
                    "option_d": "其他",
                },
            ),
        },
    )

    result = response.choices["intent"].choice
    results.append(result)

    print(f"[{index + 1}/{len(df)}] {query} -> {result}")




df["result"] = results
df.to_excel(output_file, index=False)

print(f"\n处理完成，结果已保存至：{output_file}")