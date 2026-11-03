# ReSkill-GRPO 本地移植

上游仓库：`https://github.com/aim-uofa/Self-Distillation-AgentRL`，基线提交
`35a44f5c5647b1785ad23ddf0ffbd196b695bf86`。
本实现迁移此前 SEED 中实际使用的 `seed/grpo_skill_refinement.py` 方法逻辑，运行时不依赖 SEED。
支持本项目原生同步 vLLM + FSDP/FSDP2、ALFWorld 文本环境、actor sequence parallel = 1。
不声称已验证其他 benchmark、Megatron 或异步 rollout。

## 模型和本地目录

三份模型已实际迁入本项目，而非在新项目中建立指向旧目录的模型链接：

- `models/Qwen3.5-2B`
- `models/Qwen3.5-4B`
- `models/Qwen3.5-9B`

旧 `experiment_data/seed/models/Qwen3.5-*` 路径保留指向新目录的兼容链接。
各模型的 `local-download-manifest.json` 保留上游 revision、各 shard SHA256，记录迁移路径。
本地迁移记录和重新校验结果位于 `local_assets/model-migration.json`、`local_assets/model-verification.json`。
Qwen3-4B、Qwen3-8B 及旧 Qwen2.5 实验不属于此次迁移。

`launchers/reskill_local_env.sh` 将环境、模型、数据、输出和缓存默认放在本项目下。
短路径 `../sdar-ray` 是指向本项目 `local_assets/ray` 的链接，避免 Ray Unix socket 路径过长。
模型、环境、缓存、日志不进入 Git。

## 方法与新框架的对应

1. 原生环境 reset 后，从任务描述生成一个 `<skill>...</skill>`，同一任务的
   `env.rollout.n` 条轨迹共享此 skill。检查任务、初始 observation 和 gamefile 的组内一致性。
2. 在原生 `TrajectoryCollector` 的每次动作提示中加入 episode skill；不修改环境 observation 历史。
3. 沿用原生 reward manager、GRPO advantage、KL、PPO clipping、FSDP 和 checkpoint 路径。
4. 执行结束后，用组内所有轨迹及有效动作、奖励、下一 observation 反馈生成一次贪心改写。
   过长轨迹均匀压缩摘录；保留各条轨迹的结果。无法装入上下文时拒绝该样本，不截掉任务或指令。
5. 默认要求改写格式正确且原始轨迹组中至少有一次正奖励，再形成 SFT 样本。
   **SFT 输入重新构造为最初的 task→skill 提示，不包含反馈或改写提示。**
6. 在原生 Actor 每个 PPO mini-batch 的 GRPO backward 后、同一次梯度裁剪和 optimizer step 前，
   累积 `sft_coef × mean(example 的 response-token 平均 NLL)`。没有第二次 optimizer step。
   沿用此前方法对每个 PPO mini-batch 使用本轮已接受 skill 的语义。
   rank 间补齐的 SFT 行权重为 0，按有效样本数归一化，所有 rank 的 forward/backward 次数一致。
7. 验证只按任务贪心生成执行前 skill，再执行原生验证 rollout；不反馈改写、不进行 SFT。

`require_success` 指原始 skill 的执行证据，不能证明改写后的 skill 已提升任务成功率。
`skill_refinement/*` 和 `actor/skill_refinement_*` 保留原有监控指标名，便于与旧结果比较。
每次训练 rollout 的原始/改写 skill、完整轨迹、过滤原因写到
`checkpoints/reskill_grpo/alfworld/<run>/skill_refinement/step-<N>.jsonl`，同一步重试追加记录。

入口为原生 `verl.trainer.main_ppo`，开关是 `algorithm.reskill_grpo.enable=true`。
默认关闭；其他方法不会生成 skill 或附加 SFT。无须静态 skill bank 或外部 teacher。

## 独立环境

```bash
# 安装过程在 tmux 中执行；这条命令只安装，不启动训练。
bash launchers/setup_reskill_local.sh
```

默认新环境为 `local_assets/conda/envs/agentrl-reskill`，不会更新旧 seed 环境。
沿用上游 Python 3.10、Torch 2.10、Transformers 5.13、vLLM 0.19.1 及 kernels 的版本约束。
本机默认 Anaconda channel 要求接受额外条款，因此本地入口使用 conda-forge/nvidia，
Java 使用 conda-forge OpenJDK 11；其 patch 版本可能与上游 defaults 的 11.0.27 不同。
Search retriever 的可选 faiss 安装默认关闭。
安装日志在 `local_assets/setup/`；必须确认安装退出码为 0，才视为 GPU 环境就绪。
不要使用旧 seed 环境运行 Qwen3.5 GPU 训练，它的依赖版本与新项目不同。

## 配置检查与训练入口

```bash
# 只输出配方，不加载模型、不启动 Ray 或训练，不需要进入 tmux。
MODEL_SIZE=4B bash launchers/train_reskill_alfworld.sh --print-config
```

本地配方默认沿用此前实验的主要训练设置：train=16、val batch=128、group=8、
PPO mini-batch=256、learning rate=1e-6、gamma=0.95、KL=0.01、环境最多 30 步、
prompt=2048、response=512、总训练 160 steps、每 5 steps 验证、辅助 SFT 系数 0.1。
默认设备为 0,1,2,3；**脚本不会自动运行**。默认在最终 step 保存 checkpoint。

硬件实现采用新框架的 micro-batch 默认值和 Qwen3.5 backend，不移植旧 Qwen2.5/H200 的
allocator、激活预留或 vLLM 权重驻留补丁。新的显存峰值需要另行实测，不能从旧实验推断。
使用新框架的原生 ALFWorld 样本选择；`ALFWORLD_STRICT_EVAL=0` 用于保留 128 的验证 batch，
不是上游 `PAPER_CONFIG=1` 默认的 140 个 strict seen games 协议。

获得运行实验的指示、环境与 ALFWorld 数据准备好后，可在 tmux 内使用：

```bash
source launchers/reskill_local_env.sh
# 如果本项目中尚无 ALFWorld 资产，只准备该 benchmark：
bash launchers/prepare_alfworld.sh

MODEL_SIZE=2B GPU_ID=0,1,2,3 bash launchers/train_reskill_alfworld.sh
# 或 MODEL_SIZE=4B / 9B
```

每次自动产生带 UTC 时间和随机后缀的新 experiment name；拒绝复用已存在的 run 目录。
W&B 默认 online，自动分配独立 run id；本地文件在本项目 `wandb/` 和 `local_assets/`。
凭据仅从进程环境或父目录的规则文件加载，不写入命令行、Hydra 配置或 recipe JSON。
可通过环境变量与末尾 Hydra 参数显式覆盖配置；不要把新的 recipe 当作旧实验的 resume 入口。

改写 prompt=8192、response=384，所以单独为该方法将 vLLM 上下文容量设为 9216；
动作 prompt/response 预算仍是 2048/512。配置不足时在训练初始化时直接报错。

## 合并和验证

```bash
bash launchers/merge_reskill_grpo_checkpoint.sh \
  checkpoints/reskill_grpo/alfworld/<run>/global_step_160

# 在 tmux 中执行需要 GPU 的验证；按任务生成 skill，不使用结果反馈。
GPU_ID=0 bash launchers/test_reskill_grpo_alfworld.sh \
  checkpoints/reskill_grpo/alfworld/<run>/global_step_160/merged_hf
```

验证入口复用原生 test_pretrained 基建，并启用相同 ReSkill 执行前 skill 生成路径。

## 已执行验证及边界

```bash
python -m unittest discover -s tests -p test_reskill_grpo.py -v
python -m verl.trainer.main_ppo --cfg job \
  algorithm.adv_estimator=grpo algorithm.reskill_grpo.enable=true \
  actor_rollout_ref.rollout.max_model_len=9216
```

13 项 CPU 测试覆盖格式过滤、上下文拒绝、任务提示与监督隔离、padding/位置编码、
非整除分布式样本的全局平均梯度、原生 Actor 中 GRPO+SFT 仅一次 Adam step、
原生采样接口的 metadata/长度/补齐处理、原生 rollout 训练与验证接口、无成功证据时拒绝 SFT。
三份真实 Qwen3.5 tokenizer 的非思考模板、task-only SFT 和 EOS 检查也已通过。
原生 Hydra 配置可解析。测试不启动 Ray cluster，不分配 GPU，不执行 Qwen3.5 模型 forward。
完整 Qwen3.5 GPU 训练、实际分布式 FSDP 通信和显存峰值尚未验证；本次未启动新实验。
