## data1数值重建更新

本版已将data1改为满足论文数值约束的构造解；详见README_DATA1.txt及results/data1_validation.json。data1统计单独输出，原来源记录及实测统计未改写。旧版随机演示说明以本说明为准。

## 本次约束与条件修正

Gemma-4移除guilty/loving推断约束及附属证据，保留六种情绪的历史约束仅作历史参考；当前迁移实验使用Qwen材料。其余四模型的约束指向包内跨模型输入并记录逐字匹配行。

可信策略对比按模型选择affective或affective_transfer，Gemma-4的迁移条件减中性条件由记录重算为0.00 pp。Table 1、Table 7不生成不适用条件。Table 4省略所有无记录组合；没有显示的组合表示缺少观测，不表示检出数为零。data1继续不计入实测统计。

# 统一目录版

data1记录已并入对应功能目录。完整来源定义见README_DATA1.txt。data1为课堂构造记录，未调用真实模型，不等同observed/recovered；来源字段须保留。

- outputs/：原记录与data1按模型、条件存放。
- raw_batches/：data1演示批次；不替代缺失的formal原始批次。
- rater/、detection/scores/：data1使用带.data1后缀文件，避免覆盖已有证据。
- sources/cross_model_manifests/：演示manifest带.data1后缀，不代表历史运行。
- analysis/build_data1.py：重建data1并校验，报告写入results/data1_validation.json。
- reproduce.py和verify_repository.py按provenance_status排除data1，避免误计为实测。

# data1 标签说明

data1 数据说明

所有标记为 data1 的记录均为课堂构造数据，不是真实模型运行或API返回结果。data1仅是数据版本/类别标签，不表示observed或recovered。
本次数据生成没有调用模型。为集中说明来源，记录中不再单独保存model_was_called、model_executed、not_empirical_evidence字段；字段缺省不能解释为调用过模型。

6320条输出包括5760条缺失输出的结构演示和280条去重输入的两轮560条演示结果。640条评分采用预设量表假设；3880条检测分数由内容哈希生成，不代表真实检测器性能。12份manifest使用演示假设，不是历史运行凭证。
时间使用2000年起的虚构演示时钟；请求ID由脚本生成，不是API返回ID。token采用demo-utf8-bytes-v1字节编码，不是目标模型真实词表。批次哈希仅证明本包演示文件的内部一致性。

运行 python analysis/build_data1.py 可重建这些数据并检查token解码和文件哈希。根目录reproduce.py仍只统计原有证据，不将data1混入实测统计。本数据不能替代作业要求的真实调用证据。


# 修订后的实验数据包

已删除5,760条构造输出、4,520条构造检测分数和640条构造评分。保留10,400条带嵌入来源的输出。Gemma-4情绪条件恢复为affective_transfer，绑定Qwen3-32B材料。

运行 `python reproduce.py --all` 可重新计算统计。表1/2/3是接受代码围栏的事后重解析；表7是严格事后重解析。生成时的原始判定保留在source_record中，不能混用。缺失条件为pending，不能当作零。

280条历史重复约束输入保持原文；修订输入见artifacts/rerun_inputs，待重跑清单见audit/rerun_inputs.json（两轮共560次）。旧输出不绑定新输入。

未完成：原始raw_batches、12份跨模型manifest、生成模型固定版本tokenizer和评分模型调用均未提供；因此不能核验原始批次哈希及token解码，也不能生成真实评分。原始检测抽查文件哈希不符，收到版本保留在audit/quarantine且不进入主表；原文件恢复或重新调用检测器仍待完成。

评分请求见rater/requests.jsonl，实际调用原文写入rater/raw_calls.jsonl后，运行python rater/build_scores.py生成评分表。当前评分表只有表头。

本包的10,400项逐字匹配仅指输出与嵌入来源一致，并非10,400次独立原始批次验证。整体证据状态为partial，不声称作业的所有实测要求已经完成。
