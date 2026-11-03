# ReSkill-GRPO 实现说明

## 数据流与监督边界

模型先根据任务生成初始 Skill，同一组的 G 条轨迹共享它。每一步环境观察仍然保留，并在动作提示中加入 episode-level Skill。执行结束后，改写条件包含任务、原 Skill 和组内轨迹反馈。

常规 Skill SFT 重新构建最初的任务到 Skill 提示，使用改写 Skill 作为标签，包括 XML 标记和 EOS。它不复用改写提示的 token，也不把执行反馈作为学生输入。SFT 与 GRPO 在同一次优化器更新前累积梯度，不额外执行一次独立 Adam 更新。

默认只接受格式正确、预算内且原轨迹组至少有一次成功的改写。这是成功证据筛选，不是改写效果检验：例如改写遗漏加热或冷却步骤时，即使格式合格也可能降低质量。

## 当前生成提示词

```text
Before solving the task, write a concise, task-specific procedural skill. Describe a plan, pitfalls, and checks; do not execute actions or invent observations. Do not output analysis, reasoning, or thinking tags. Output only your skill enclosed in <skill>...</skill>, with no text outside these tags.
Task:
{actual task text}
```

## 当前改写提示词

```text
Analyze the completed trajectories and their environment feedback. Rewrite the initial skill to improve future execution of this task. Use successful behavior and correct observed mistakes. Only state procedures usable BEFORE execution; do not invent observations, copy an answer, or output a replay of one trajectory. Do not output analysis, reasoning, or thinking tags. Output only the revised skill enclosed in <skill>...</skill>, with no text outside these tags.
```

这段指令后附加实际任务、初始 Skill 和轨迹反馈。过长轨迹按预算摘录，各次执行的结果仍然保留。

## 可选重写质量监督

代码保留周期性 Rewrite Quality Loss，当前 4B 默认关闭。启用时，每隔指定步数，对原 Skill 估计成功率低于 0.75 的组进行配对改写评估；仅当新成功率严格提高且归一化内容存在差异时，给予改写条件下的正向 SFT，否则使用 Unlikelihood Loss。

被判为无效的样本会从常规 Skill 正向 SFT 中屏蔽；没有符合条件的任务时不产生这项辅助梯度。有限采样只提供质量估计，不保证改写一定更好。其输入条件与常规任务到 Skill 的 SFT 不同，不能混为一项监督。

## 训练与验证记录

原训练器在每一步保存初始和改写 Skill、轨迹、筛选原因及相关指标；验证时只生成执行前 Skill 并执行环境，不根据验证结果改写或反向传播。验证 Skill 日志记录实际任务实例、生成文本、格式、回合奖励及步数。

成功率来自环境 `won`。训练成功率是当前任务批次的执行成绩，不是同一固定评测集的学习曲线；周期性 `valid_seen` 验证也存在重复抽样和样本组合变化。比较 checkpoint 时，应区分采样验证、固定实例评测和未见场景评测。
