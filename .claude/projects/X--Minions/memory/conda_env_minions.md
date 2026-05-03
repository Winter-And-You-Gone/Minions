---
name: conda环境Minions
description: 当前项目所有Python相关操作必须在conda Minions环境中运行
type: feedback
---

所有 Python 相关命令（pip install、python、pytest 等）必须通过 `conda run -n Minions` 或先 `conda activate Minions` 后执行。
不要在系统全局 Python 或其它环境中运行此项目的代码。

环境信息：
- 环境名：Minions
- Python 版本：3.12
- 位置：F:\Anaconda\envs\Minions

**Why:** 用户明确要求创建独立的 conda 环境隔离此项目的 Python 依赖。

**How to apply:**
- 使用 `conda run -n Minions python ...` 运行任何 Python 脚本
- 使用 `conda run -n Minions pip ...` 安装任何 Python 包
- 使用 `conda run -n Minions python -m pytest ...` 运行测试
- 不要在命令中直接使用裸 `python` 或 `pip`
