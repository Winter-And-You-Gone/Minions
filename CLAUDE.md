# Minions 项目规则

## 必须严格遵守以下规则

### 1. 使用中文回答
所有面向用户的文本必须使用中文，除非是技术专有名词或行业通用缩写（API、CLI、JSON、YAML等）。代码中的标识符、关键字、报错信息保持原文。

### 2. Python 命令必须使用 conda Minions 环境
所有 Python 相关操作（python、pip、pytest 等）必须通过 `conda run -n Minions` 执行：
- `conda run -n Minions python ...`
- `conda run -n Minions python -m pytest ...`
- `conda run -n Minions pip install ...`

禁止直接使用裸 `python` 或 `pip`。

### 3. 改代码后必须编译、提交、同步
每次修改代码后，自动执行以下流程：
1. **编译/测试** — 运行 `conda run -n Minions python -m pytest tests/` 确保通过
2. **提交** — 通过后自动 git add → commit
3. **同步** — 自动推送到 GitHub 远程仓库

不需要等用户提醒才推送。

### 4. 项目信息
- Python 3.12, conda 环境名: Minions
- 包管理器: pip (pyproject.toml)
- 配置文件: config.yaml
- 测试框架: pytest
- 主入口: `python -m voice_agent.main`
