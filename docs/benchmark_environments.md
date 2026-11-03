# 四个独立 Conda 运行环境

环境均位于本项目 `local_assets/conda/envs/`，已加入用户本地 Miniconda 的 `envs_dirs`。
不替换旧 `seed`、`seed-webshop`、`seed-retriever` 环境，也不启动训练。

| Conda 名称 | 用途 | 主要运行组件 |
|---|---|---|
| `agentrl-alfworld` | ALFWorld 文本环境 | Python 3.10、ALFWorld 0.4.2、TextWorld 1.7.0、Fast Downward、CPU Torch 2.10 |
| `agentrl-webshop` | WebShop 原生 Python/Lucene 环境 | Python 3.10、Pyserini 0.17、OpenJDK 11、spaCy 3.8.7、CPU Torch 2.10 |
| `agentrl-searchqa` | SearchQA 数据和 E5 检索 | Python 3.10、Transformers 5.13、Torch 2.10 CUDA 构建、FAISS CPU 1.12 |
| `agentrl-sciworld` | ScienceWorld 模拟器 | Python 3.10、ScienceWorld 1.2.3、Py4J 0.10.9.9、OpenJDK 11、CPU Torch 2.10、Ray 2.50 |

FAISS 使用 CPU 后端；E5 模型可以使用 CUDA，实际启动检索服务时再按需求选择设备。
ALFWorld/WebShop 的环境进程无需加载 Qwen/vLLM，因此使用 CPU Torch，不占用训练显存。
这四个环境不是完整的 Qwen3.5 训练环境；训练依赖和各 benchmark 的运行依赖分开管理。

## 激活

```bash
source /workspace/fudan/donghaosen/miniconda3/etc/profile.d/conda.sh
conda activate agentrl-alfworld
# 或 agentrl-webshop / agentrl-searchqa / agentrl-sciworld
```

也可从项目目录运行：

```bash
source launchers/benchmark_envs/activate.sh webshop
```

各环境的 Conda 激活 hook 设置项目路径、数据路径与缓存，隔离容器预置的 Python 3.12 Torch
动态库，防止它污染 Python 3.10 环境；deactivate 时恢复原有变量。

## 数据、缓存和日志

遵守共享机器目录规则：软件环境和脚本位于 `/workspace/fudan/donghaosen`；
新下载数据、归档、大缓存和索引位于：

```text
/data/fd/donghaosen/Self-Distillation-AgentRL/
  data/alfworld/
  data/webshop/
  data/webshop-lucene/
  data/searchR1/
  data/searchR1_processed_direct/
  data/scienceworld_agentgym/
  models/e5-base-v2/
  downloads/
  cache/
  logs/
```

项目 `data/`、WebShop 原生数据和索引路径、E5 模型路径均通过链接指向上述位置。
此前用户要求迁入项目的 Qwen3.5 权重保持原位置。

- ALFWorld：复制已有数据，训练 3553、valid_seen 140、valid_unseen 134 个游戏。
- WebShop：下载原始 1k 和完整商品/属性文件、人工指令；按原生文档格式构建 Lucene
  100、1k、100k 和完整索引。
- SearchQA：固定上游 revision，下载完整 E5 Flat 索引两部分、Wikipedia 2018 语料、
  E5-base-v2 和 NQ/HotpotQA 原始数据。转换复用本项目原生 `process_single_row`。
  原始下载和分块归档保留，所有 LFS 文件核对 SHA256，语料解压核对 gzip CRC。
- ScienceWorld：使用原生 AgentGym 固定 revision/校验和与任务编号协议；原始训练池 2120，
  排除与测试重叠的 61 个 ID 后训练集 2059，测试集 200。

## 检查与安装记录

包版本快照和实际检查结果在 `/data/fd/donghaosen/Self-Distillation-AgentRL/logs/`：

- `<benchmark>-final-pip-freeze.txt`：安装版本快照。
- `<benchmark>-smoke.json`：已实际执行的检查及其范围。
- `environment-summary.json`：四个环境的安装和检查汇总。

独立执行检查：

```bash
source launchers/benchmark_envs/activate.sh alfworld
python launchers/benchmark_envs/smoke.py alfworld
```

ALFWorld 检查原生文本环境 reset/step。WebShop 检查 spaCy、Lucene、reset 和搜索动作。
SearchQA 检查真实 E5 CPU forward、FAISS 查询、数据表和语料，另行验证完整生产索引。
默认 ScienceWorld 仅检查包和模拟器 JAR 完整性，不启动 Java 网关。

`AGENTS.md` 禁止监听端口，因此 HTTP 检索服务不会自动启动；ScienceWorld 仿真器
reset/step 需要用户明确允许本机 Java/Py4J 临时监听端口后，才能使用
`python launchers/benchmark_envs/smoke.py sciworld --allow-local-port`。

安装入口和各环境依赖列表保存在 `launchers/benchmark_envs/`。
以实际退出码、`pip check` 和 smoke JSON 为准，不将“后台任务已启动”视为“安装完成”。

## 本次验收结果

四个环境的 `pip check` 均通过，且均可导入本项目对应的原生环境适配模块。
ALFWorld、WebShop 和 ScienceWorld 实际 reset/step 检查通过。
ScienceWorld 自检使用了用户明确允许的临时本机 Java/Py4J 端口，检查后已关闭并确认无残留网关进程。
SearchQA 已加载完整 768 维生产索引，核对索引与语料均为 **21,015,324** 条，
完成真实 E5 查询与 FAISS 检索；未启动 HTTP 服务，所有检查不占用 GPU。
WebShop 同时保留 1k、100k 与完整商品索引；原生 Lucene 对无可索引文本的文档跳过，
计数在各索引的 `local-index-manifest.json` 中记录，不把正常的 empty 文档当作索引错误。

最终 Conda 版本快照在 `launchers/benchmark_envs/locks/<benchmark>/environment.yml`，
最终 pip 快照在数据目录 `logs/<benchmark>-final-pip-freeze.txt`。
这些是已安装版本记录，不替代所有 pip 包的完整哈希锁定。
本次未启动任何新训练任务。
