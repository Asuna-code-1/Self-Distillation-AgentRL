# 方法代码入口

这里通过相对链接索引核心实现，实际代码位于 `framework/` 的原内部路径。保持这些路径是为了不改导入方式和运行时相对路径，也避免复制两套实现。

- `reskill_grpo.py`：初始 Skill、改写、筛选及常规 SFT。
- `rewrite_quality.py`：可选的周期性重写质量监督。
- `numerics.py`：跨 rank 数值检查。
- `gpu_resource_trace.py`：分阶段显存记录。
- `packed_hybrid.py`：Qwen3.5 打包序列边界辅助。

联合优化接入仍在 `framework/verl/workers/actor/dp_actor.py`；训练与环境衔接仍在原 Trainer 和 TrajectoryCollector 中。
