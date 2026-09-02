# PanelTone alpha.3 执行索引

本索引对应仓库外的持久计划副本：`F:\AIALRA PanelTone Data\plans\paneltone-alpha3-plan-20260902.md`。它只记录公开的执行边界，不保存业务数据、凭据或恢复档案。

- 目标：`0.2.0-alpha.3`；唯一 `main`、唯一工作树；不创建分支、PR、Release 或 Docker。
- live 边界：4 个任务、244 页和孤立 `jobs/417203` 必须保留；不删除源图、生成图、数据库或旧结果备份。
- 批次：几何锁定上色与旁路/QA/事务 repair → WebP 预生成/预取/下载/日志 → 响应式/进度/导航 UI。
- 发布：每批独立测试和轻量敏感扫描；fetch 后确认远端未变；普通 fast-forward 到 `main`，不强推。
- 复核命令：`git status --short --branch`、`git worktree list`、`git log -1 --oneline`，以及本文件和外部计划副本。
- 当前状态由实际命令输出证明；没有证据的服务、线上登录、人工体验和 live repair 结果不得宣称完成。
