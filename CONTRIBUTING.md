<div align="center">
  <h1 align="center">参与 PanelTone 开发</h1>
</div>

## 1. 开发范围

欢迎修复稳定性、可访问性、导入安全、任务恢复、确定性合成与文档问题；模型权重、未授权漫画页面、真实任务数据库和本机日志不能进入提交

## 2. 本地检查

提交前依次运行：

```powershell
# 检查 Python 代码并自动修复安全的格式问题
.\.venv\Scripts\python -m ruff check src tests --fix
.\.venv\Scripts\python -m ruff format src tests

# 运行完整 Python 测试，其中包含 300 页合成压力测试
.\.venv\Scripts\python -m pytest -q

# 编译用户界面，确认 TypeScript 类型和生产构建通过
Set-Location src\manga_repaint\web\frontend
npm ci
npm run build
```

## 3. 提交要求

- 使用合成数据复现问题，不能上传真实漫画或用户文件
- 新接口需要测试成功、失败和恢复路径
- 新选项需要用途、适用场景、改变内容与主要代价说明
- UI 改动需要检查 `1440×900`、`1024×768` 与 `390×844`
- 新模型只提交清单、固定修订号、下载过滤规则和许可证链接

## 4. 安全问题

文件读取越界、压缩包逃逸、远程暴露、任意代码执行、秘密泄露和不安全删除请使用当前仓库 Security 页面的 GitHub 私密安全报告功能，不要先创建公开 Issue
