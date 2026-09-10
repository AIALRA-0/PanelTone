# 分区底色与材质渲染

该模块是可执行的隔离验证路线，不声称已经解决任意漫画的全自动材质识别

生产边界固定为

- 原图拥有几何、线条、文字、气泡、格线和明暗结构
- Cobra 和 FLUX 只能提供色彩建议
- SAM2 只能提供待审核的几何分组
- VLM 只能为已编号区域提议标签，不能输出坐标或直接写像素
- 只有确定性渲染器可写入可发布成品
- 未知区域、冲突、未通过 QA 或未审核页都会阻止发布

## 已实现的数据流

1. \`SegmentGraph\` 保存与源图等尺寸的原子区域标签、邻接关系和未知比例
2. \`BookArtBible\` 保存角色、外观、服装、场景、色槽、区域事实、证据和版本状态
3. \`MaterialPlan\` 保存像素标签、保护区、材质绑定和配色版本，零标签只表示未知
4. \`render_material_layers\` 输出平涂、阴影、网点和最终层
5. 普通网点只参与明暗层级，不会生成新色相
6. 只有明确标为 \`patterned\` 的材质可保留弱无色纹理
7. \`PublicationManifest\` 联合检查艺术圣经、区域图、QA 与人工复核证据

## 隔离命令

\`\`\`powershell
paneltone flat-proposal source.png candidate.png review-flat
paneltone sam2-flat-proposal source.png candidate.png review-sam --model-root sam2-model
paneltone material-render source.png review/plan.json review-render
paneltone cel-candidate source.png candidate.png review-cel
\`\`\`

所有输出目录必须不存在，这些命令不构造正式任务管理器，不恢复任务，不写 live 数据库，不替换正式图片

\`cel-candidate\` 可叠加高分辨率小格色彩建议

\`\`\`powershell
paneltone cel-candidate source.png page-candidate.png review-cel \`
  --refinement 770,0,1218,920=small-panel-candidate.png
\`\`\`

它会产生 \`albedo-proposal.png\`、\`cel-locked.png\`、\`display.webp\` 和 \`review.json\`，清单始终标记 \`publishable=false\`

## 渲染与网点

- 源图分解为保护像素、结构墨线、宽变明暗和高频网点
- 每个已审核区域只引用一个固有色槽
- 阴影从源图宽变明暗和受限场景光计算
- 候选图的 RGB、纹理、高光、边缘和几何不进入已审核材质渲染器
- 白色衣物、纸白、高光和未上色区域必须显式区分

## 自动验收

- 保护像素逐像素一致
- 源图、标签图、配色计划和渲染结果都有可复查哈希
- 未知、重叠冲突、缺失色槽和过期依赖阻止发布
- 单一固有色区域检查色相离散、低频色斑、中性孔洞、边界染色和网点色相耦合
- 几何 QA 与语义 QA 分开，不用渲染器自己的标签假装成语义真值

## 当前发布门禁

- 已完成 12 个高风险页的隔离黄金候选
- 候选可在网页的「专业平涂」中查看
- 小漫画格仍存在未着色人物，实图 QA 会拒绝该页
- 244 页 live 不会在黄金样本未经用户确认时自动替换

参考：[SAM 2 官方仓库](https://github.com/facebookresearch/sam2)、[Split Filling](https://lllyasviel.github.io/SplitFilling/)、[FlatMagic](https://cragl.cs.gmu.edu/flatmagic/)、[ShadowMagic](https://cragl.cs.gmu.edu/shadowmagic/)
