# 目录整理说明

本次是文件组织调整，不是训练代码重构。整理前提交为 `4ce2da4cdb879259b95b22b6de9bda866c014f76`。

| 整理前 | 整理后 |
|---|---|
| `verl/` | `framework/verl/` |
| `agent_system/` | `framework/agent_system/` |
| `gigpo/` | `framework/gigpo/` |
| `launchers/` | `framework/launchers/` |
| `tests/` | `framework/tests/`，根目录提供快捷入口 |
| 根目录依赖与包配置 | `framework/` 下原文件名 |
| 原 README 与 docs 文档 | `docs/reference/`，保留历史内容 |
| 核心方法文件 | 实际位置保持在框架内部，`method/` 提供快捷入口 |
| 本机实验脚本 | 实际位置保持在框架内部，`experiments/launchers/` 提供快捷入口 |

完整逐文件移动关系及 SHA256 保存在 [layout-map.json](layout-map.json)。其中的历史源目录、提交和文件摘要用于核对来源，不包含模型或运行日志。

链接均为仓库内部的相对软链接，目标文件随仓库一起提交。目录调整没有修改已有 Python、Shell、YAML 配置文件的内容。运行时仍从 `framework/` 使用原来的 `verl`、`agent_system` 等导入路径。活动训练使用的原工作目录独立于这次 GitHub 整理，没有被搬动或覆盖。

## 来源与保留范围

基础仓库是 aim-uofa/Self-Distillation-AgentRL，基线提交 `35a44f5c5647b1785ad23ddf0ffbd196b695bf86`。其基础设施进一步使用 veRL、verl-agent / GiGPO 等组件。原有许可证和源码版权头保留。

`framework/` 含共享训练基础设施、其他方法和环境的既有实现。此次不大幅删除这些文件，避免破坏内部依赖；它们不作为本项目的方法贡献宣传。ReSkill-GRPO 的方法入口和实验说明单独展示在仓库首页与 `method/` 中。

原文档仅作为历史参考，其中的旧步数、旧目录或早期验证状态不代表当前实验配置；当前配置以首页索引和实际启动参数为准。

## 补齐之前遗漏的源码

先前源码打包把所有名为 `models` 的目录当作权重目录过滤，误漏了 `verl/models` 和 WebShop 的两个 Python 模块。本次从原工作目录原样补齐这些源码，以及 `packed_hybrid.py`，没有改写实现。权重文件仍不上传。补齐文件和 SHA256 单独记录在 `layout-map.json` 中。
