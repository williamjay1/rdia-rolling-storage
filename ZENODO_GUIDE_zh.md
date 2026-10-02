# 将完整公开版本上传 Zenodo，并取得 DOI

## 当前推荐归档版本：v1.1.0

请下载 https://github.com/williamjay1/rdia-rolling-storage/releases/tag/v1.1.0
中的 **rdia-rolling-storage-v1.1.0.zip** 和 **SHA256SUMS**，由你本人上传
Zenodo。这个完整包已合并核心澳大利亚派生数据和 v14/v15/v16 增补；
v1.0.1 只有绘图代码标签，不能替代完整数据包。不要选择 GitHub 自动生成的
Source code ZIP 作为完整复现数据。包内 release_manifest.json 是当前文件
清单，CITATION.cff 与 .zenodo.json 均对应软件 v1.1.0。软件标题与当前论文
主线一致，但软件 DOI 与论文 DOI 是不同对象。

上传 ZIP 与校验文件，作者填 Junjie Zhang，核对下文的单位、ORCID、混合许可
和官方来源；发布后把实际版本 DOI 与记录链接发回，以便回填论文和 CFF。
已有 DOI 若对应更早记录，使用 Zenodo 的新版本功能，不修改已存文件假冒
同一版本。本次仅整理 GitHub 公开版本，没有替你登录或创建 Zenodo DOI。


唯一作者：**Junjie Zhang**；两项单位均为 **Shanghai International Studies University**，分别为 **Shanghai Academy of Global Governance and Area Studies**、**School of Economics and Finance**，Shanghai 201620, China。邮箱：<junjiezhang2024@shisu.edu.cn>；作者已提供 ORCID：[0009-0004-8821-4018](https://orcid.org/0009-0004-8821-4018)。公开仓库为 <https://github.com/williamjay1/rdia-rolling-storage>。本指南不创建 Zenodo 记录、不预造 DOI；由作者登录自己的 Zenodo 账户完成发布。

## 推荐：手动上传完整 Release 包

1. 打开仓库的 **Releases**，下载拟归档版本的**完整发布 ZIP**及其校验清单。完整包应同时包含该版本代码、较大的澳大利亚派生输入和参考轨迹、结果与图源数据、复现说明、来源及许可通知。GitHub 自动生成的 **Source code (zip)** 只反映 Git 仓库内容，不能代替另附的科学数据 ZIP。核对 Release 标签、提交编号、文件清单和 SHA-256 校验值。
2. 阅读包内 [DATA_LICENSE.md](DATA_LICENSE.md)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和复现说明。澳大利亚派生输入保留 AEMO 条款。IESO 原始文件、含源价格的 Ontario 对齐输入及逐期轨迹不应上传；该部分通过官方链接、固定版本校验值和重建代码提供。不要自行把本机原始仓库打包加入。
3. 登录 <https://zenodo.org/>，选择 **New upload**，上传完整发布 ZIP和外层校验文件。也可以分别上传代码 ZIP 与数据 ZIP，但须确保文件清单完整且对应同一版本。Zenodo 当前常规额度为每条记录 **100 个文件、合计 50 GB**；若超额，先调整归档结构或按官方流程申请额度。[上传说明](https://help.zenodo.org/docs/deposit/create-new-upload/)
4. 填写元数据：标题与公开仓库/该版本一致；版本使用实际 Release 版本；资源类型可选择 **Software**，在描述中说明包含派生数据与复现材料；作者填 `Junjie Zhang`，录入上述两项 SISU 单位，并填写作者已确认的 ORCID `0009-0004-8821-4018`。不填写未确认基金或不存在的论文 DOI。说明澳大利亚输入的源条款、Ontario 官方重建边界及实际复现范围，并把 GitHub 仓库和对应 Release 作为相关链接。上传研究支持材料时，不能把文章 DOI 当作本条记录 DOI。[元数据与 DOI 说明](https://help.zenodo.org/docs/deposit/create-new-upload/)
5. **修改许可字段**：不要保留默认 CC BY 4.0 作为全包许可。声明原始代码 MIT、Nature QA 脚本 Apache-2.0、AEMO 来源材料按供应方条款。界面若提供 **Other (Open)**，配合包内混合条款说明使用；也可点 **Add custom**，名称填 `RDIA mixed licenses and source terms`，说明按 `DATA_LICENSE.md` 逐类适用，并链接该 Release 版本的文件。官方支持自定义及混合许可，不能靠选择一个统一 CC 许可改变第三方权利。[许可操作指南](https://help.zenodo.org/docs/deposit/describe-records/licenses/)
6. 保存草稿并 **Preview**。逐项核对作者、版本、许可、GitHub 链接、文件大小、下载文件清单及校验值，确认没有源数据禁入项、凭据或本机配置；再由作者点击 **Publish**。只有发布后的实际记录才可声称已归档并拥有注册 DOI。若使用 Reserve DOI，须在发布后确认记录和 DOI 实际可解析。[发布步骤](https://help.zenodo.org/docs/deposit/create-new-upload/)

## 发布后怎样引用与更新

首次发布通常产生两个标识：**版本 DOI** 指向这次具体归档，**概念 DOI** 代表各版本系列。论文复现说明优先使用此次**版本 DOI**；项目首页需要指向更新系列时可用概念 DOI。之后修改科学代码、输入或结果，应建新版本并核对新 DOI；不要自行拼接 `.v1`、`.v2` 来制造版本 DOI。[DOI 版本规则](https://support.zenodo.org/help/en-gb/1-upload-deposit/97-what-is-doi-versioning)，[官方引用说明](https://support.zenodo.org/help/en-gb/25-citations/234-how-to-share-or-cite-a-zenodo-record)

将实际**版本 DOI、Zenodo 记录 URL、Release 标签、提交编号和包的校验值**保存下来，再更新仓库 `CITATION.cff`、`.zenodo.json` 的适当引用字段及论文的数据/代码可用性声明。核对 DOI 指向的文件确为论文使用的版本。Zenodo 当前上传指南允许发布后 45 天内编辑文件，因此 DOI 本身不替代版本冻结或内容校验；本项目采用保留已发布包及哈希、科学内容改动另建新版本的做法。[当前文件管理说明](https://help.zenodo.org/docs/deposit/create-new-upload/)

## 可选：连接 GitHub 自动归档

在 Zenodo 账户的 GitHub 设置中连接账号、同步仓库并启用该仓库，然后创建新的 GitHub Release。官方流程对**启用后的新 Release**进行归档；已有发布版本若未被归档，推荐按上述步骤手动上传。[连接仓库](https://help.zenodo.org/docs/github/enable-repository/)，[创建并检查归档](https://help.zenodo.org/docs/github/archive-software/github-upload/)

自动归档后，必须打开实际 Zenodo 记录检查文件清单和下载内容。**不要假定 Release 附件中的大数据 ZIP 已被接收**；源代码归档与附件数据是不同对象。若缺少科学数据附件，尚不能称为完整研究归档，可改为另建完整手动记录或按 Zenodo 允许的方式补齐并重新核验。GitHub 发布附件应分别小于 **2 GiB**；本项目通过 Release ZIP 分发较大派生数据，不使用 Git LFS 指针作为存档数据。[GitHub Release 附件限制](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

GitHub 集成同时看到 `.zenodo.json` 和 `CITATION.cff` 时，**`.zenodo.json` 优先，CFF 不被用于这一集成的元数据提取**；CFF 仍服务于 GitHub 的引用入口。检查归档元数据时应以实际记录为准，不能把仓库中两份配置存在当作导入成功。[Zenodo JSON 规则](https://help.zenodo.org/docs/github/describe-software/zenodo-json/)，[CFF 支持范围](https://help.zenodo.org/docs/github/describe-software/citation-file/)

本版本保留 v10 核心科学冻结结果和 v11 图形设计，并将 v13 追加诊断另行标识，不能把版本整理说成整套原始分析的新复现。数值区间与检验均以已保存的程序、选择规则及观察历史为条件。128期 NSW 求解、30个 SPO+ oracle validation cases、已保存路径统计重建、7图重绘和 Ontario 两臂输入准备，各有具体执行范围；不能合并称为全部原始管线的独立外部复现。Ontario A/B/C三层完整诊断未执行的部分也不能声称验证完成。

新增30分钟可用性延迟已在研究项目中完成20次回放，每次46,741个origin，保留原2023年C*权重和共同时间顺序。五资产S−C*变为−A$3,944.46，IL−C*仍为+A$57,139.02，覆盖974个名义日。这是点敏感性结果，没有新增置信区间，也不是观测到的参与者接收时间。新增文件见 `provenance/v13_action_files.json`；不能把源项目完成的20次计算改称公共入口已经全网格重新运行。公共入口的固定子样本动作范围和SA证书已实际通过，包含750行比较、1,125次全二元范围计算及与原结果的精确比对，见 `provenance/v13_action_validation.json`；这不等于重新求解全部7,305个共同状态或全20次延迟网格。

上述归档操作不会重算研究结果，也不证明期刊已接受论文。未投稿主文、内部审稿意见和作者私人文件不上传。归档范围、复现范围和第三方数据边界应与实际包内容一致。官方操作说明核验日期：**2026-10-02**。
