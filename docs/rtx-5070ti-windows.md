# RTX 5070 Ti Windows 使用说明

**简体中文** · [English](rtx-5070ti-windows.en.md) · [可执行包下载与安装](rtx-5070ti-windows-downloads.md)

这套版本面向 **RTX 5070 Ti 16 GB、Windows x64、单模型单并发**。使用 CUDA 13.4.2、
Native SM120a Release 引擎和 **NInfer Manager 1.3.1** 托盘管理器。日常只需双击
`NInferManager.exe`，模型管理和性能监控都在浏览器中完成，无需另外运行 Python、
CMD 或 PowerShell 脚本。

默认提供 **XXS 160K、S 128K** 两个配置，均使用 `strict` 显存策略。K 表示
1024 tokens。本文说明怎么使用、文件放哪里、参数含义，以及实际测过什么。

## 1. 先启动管理器

1. 将完整运行包解压到固定目录，例如 `qwen27b`。不要只复制 EXE。
2. 把 `.ninfer` 模型主文件及其全部分卷放入 `model/`。下载包不含模型权重。
3. 双击 `NInferManager.exe`。它没有主窗口或终端窗口，图标出现在 Windows 托盘中。
4. 右键图标，选择“管理模型”检查配置；从“启动模型”的二级菜单选择 XXS 或 S。
5. 就绪后打开监控，或从托盘复制 API base 和模型名供客户端使用。

| 模型已就绪 | 模型停止或尚未就绪 |
|---|---|
| ![运行中的绿色 N 托盘图标](assets/ninfer-tray-running.png) | ![停止状态的灰色 N 托盘图标](assets/ninfer-tray-stopped.png) |
| 浅绿色底、深绿色字母 N | 灰色图标；加载、停止中和异常状态由提示文字区分 |

默认管理页面是 `http://127.0.0.1:8090`，推理 API base 是
`http://127.0.0.1:18081/v1`。从托盘打开页面会建立本地管理会话。
XXS 和 S 对外都使用模型名 **`swift-1.5-qwen3.8-27b`**，一次只加载一个。
启动、运行、停止期间禁用再次启动，避免重复加载。

关闭浏览器不会停止模型；停止模型后管理页面仍可用；退出管理器会停止它启动的模型和
监控网站。Windows Job 绑定模型及其子进程，正常停止先发送 CTRL_BREAK。管理器不会
仅凭相同端口或 PID 接管其他进程。

### 随 Windows 启动如何实现？

设置里的“随 Windows 启动”使用当前用户注册表
`HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run` 下固定的
`NInferManager` 项。
Windows 在这个用户**登录后**启动管理器，不需要管理员权限。它不是登录前运行的系统
服务；重启后停留在登录界面时不会提前加载模型。

| 独立设置 | 开启后的行为 |
|---|---|
| 随 Windows 启动 | 用户登录后启动托盘管理器 |
| 自动启动默认模型 | 管理器启动后加载默认配置，出厂默认是 `xxs-160k` |

首次默认**关闭随 Windows 启动**，需要在网页或托盘手动开启；默认模型配置仍启用自动
加载 XXS 160K。已有个人设置继续使用自己的值，不会因为升级重新开启这两项。是否随
Windows 启动以实际登记和 Windows 的启动应用状态为准。

一台电脑可以放多份管理器，但当前用户只有这一个自启登记：**最后显式开启或保存自启
设置的那份程序生效**，会覆盖该项中的启动路径。普通双击或 `--autostart` 启动本身不会
把登记抢回旧副本。关闭旧副本的自启时，如果该项已指向另一份程序，不会删除新副本的
登记。移动程序后，从新位置显式开启自启即可更新路径。

Windows 的“启动应用”可以另外禁用此项，管理器尊重该禁用状态；设置文件里写着开启，
不代表 Windows 一定会执行。需要恢复登录启动时，也要在 Windows 中允许该启动项。

此前 1.3.0 曾实际执行登记命令，自动加载 XXS 160K 并正确返回短回复，重复执行也未重复
加载模型。那是旧版本的命令链路验证，不是 1.3.1 新自启规则的实测结果。**尚未通过真正
重启并重新登录验证 Windows 本身的触发时机。**

## 2. 文件分别放在哪里

1.3.1 将**程序目录**和**个人数据目录**分开。EXE、引擎、模型和网页留在程序目录
（PackageRoot）；生效配置、日志和运行状态放在数据目录（DataRoot）。正常安装在
`Program Files` 时不需要向程序目录写入，也不需要为日常运行授予管理员权限。

默认数据目录是 `%LOCALAPPDATA%\NInferManager\<安装位置ID>`，ID 根据程序所在目录生成。
不同安装位置使用不同的子目录。第一次使用某个位置时，先复制该程序包的 `config/`，
再用内嵌默认资源补齐缺少的首次配置；之后只读取数据目录中的生效配置，不会用包内
文件反复覆盖修改。

只有默认数据目录无法创建或无法写入时，才回落到程序目录保存数据。配置 JSON 损坏
会直接报错并保留，不会借此切换目录或悄悄恢复默认值。

```text
程序目录 / PackageRoot，例如 qwen27b/ 或 Program Files 下的安装目录
├─ NInferManager.exe               托盘管理程序，Release / Windows x64
├─ engine/
│  ├─ ninfer-serve.exe              CUDA 13.4.2 Native SM120a Release 引擎
│  └─ *.dll                        引擎依赖，保持与 EXE 放在一起
├─ model/
│  ├─ *.ninfer                     模型主文件
│  └─ …                            全部续卷，保持原文件名
├─ config/                         首次初始化用的配置种子，不是日常生效副本
├─ wwwroot/                        管理和监控网页的正式构建文件
├─ docs/                           中文说明、英文说明、下载页及图片
├─ LICENSE                         项目许可
└─ licenses/                       NVIDIA EULA、依赖许可及来源

数据目录 / DataRoot
%LOCALAPPDATA%\NInferManager\<安装位置ID>/
├─ config/
│  ├─ settings.json                语言、自启、默认配置、扫描目录、管理端口
│  ├─ profiles/
│  │  ├─ xxs-160k.json              XXS 的路径、API 名称和启动参数
│  │  └─ s-128k.json                S 的路径、API 名称和启动参数
│  ├─ chat_template.jinja           当前会话模板
│  ├─ chat_template.LICENSE         模板许可
│  ├─ device-profiles.json          当前 GPU 校准结果
│  ├─ initialized.json             已完成首次初始化的标记
│  └─ history/                     保存配置前的历史副本
├─ runtime/                        本机运行状态和短期会话信息
└─ logs/                           引擎输出、错误和请求指标
```

数据目录中的 `config/` 是需要备份保留的个人配置；`runtime/` 是运行时数据，不是要分享
的配置。网页保存的参数和 `config/history/` 历史都在数据目录。
可执行发行包不包含个人历史、日志、会话凭证和模型权重。旧引擎、测试脚本、编译工具、
原始 GGUF 和基准日志都不需要放在日常运行目录。

网页保存的参数在**下次启动模型**时生效。保存采用原子替换，原件归档到
`config/history/`；损坏的 JSON 会报错并保留，不悄悄用默认值覆盖。

相对路径也分清用途：`config/chat_template.jinja`、`config/device-profiles.json`
等配置资源从数据目录解析；`engine/ninfer-serve.exe`、`model/...` 和模型扫描目录从
程序目录解析。绝对路径仍指向指定位置。

移动或复制程序到新位置会得到一个新的数据目录，首次从该包的 `config/` 初始化；以后
两个位置各自保存参数，互不覆盖。移动程序文件不会自动搬走原数据目录中的后续修改，
需要保留这些修改时应先备份原数据目录的配置。仓库默认资源位于
`apps/windows-manager/config/`，同时内嵌在管理器中；包配置与数据配置不会持续双向同步。

### 增加模型与修改参数

网页可编辑扫描目录、模型路径、对外模型 ID、启动参数和默认配置。默认扫描
`model/`，启动时和每 15 秒刷新，也可手动刷新。扫描只检查 v3 容器头、目录和分卷
完整性，不读取全部权重或初始化 GPU。GGUF、safetensors、单独续卷不是可启动入口。
“文件完整”不代表任意上下文都能装入显存。

参数旁的问号支持鼠标悬停和键盘聚焦。右上角可切换简体中文和 English，语言会保存并
同步托盘菜单，不会改变模型参数或丢弃尚未保存的编辑。

## 3. 默认模型和参数

模型是 `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.ninfer` 和对应的
`IQ3_S-mtp.ninfer`：从包含多种张量量化类型的 GGUF 转换为 text/MTP v3 容器，**没有可用的图片
理解组件**。原始 GGUF 和 conversion report 不是运行依赖；换模型文件需重验容量和速度。

| 项目 | XXS 默认 | S 默认 |
|---|---|---|
| 配置 ID | `xxs-160k` | `s-128k` |
| 上下文和固定 KV 容量 | 163840 / 160K | 131072 / 128K |
| Prefill chunk | 1024 | 256 |
| 并发 | 1 | 1 |
| KV / GDN state | rk8v4 / FP16 | 同左 |
| 草稿 | MTP 2、ngram 31、完整 MTP attention window | 同左 |
| Graph allowance | 72 MiB | 同左 |
| `--lm-head-draft` | 关闭 | 关闭 |
| CPU 上下文缓存 | 6144 MiB，设备快照 1 个 | 同左 |
| CUDA 显存策略 | `strict`，等于 `strict-64-128` | 同左 |
| 默认输出上限 | `--default-max-tokens 0` | 同左 |
| 采样 | temperature 1、top-p 0.95、top-k 20、min-p 0 | 同左 |
| 惩罚、种子 | presence/frequency 0、seed 42；重复惩罚保持中性 1 | 同左 |
| 思考 | 默认开启、xhigh、保留思考 | 同左 |

完整源码参数位于仓库 `apps/windows-manager/config/profiles/xxs-160k.json` 和
`apps/windows-manager/config/profiles/s-128k.json`；运行时以数据目录的 `config/profiles/`
为准。

输出上限 0 不添加固定的默认 token 上限，客户端仍可指定自己的限制。它不会扩大上下文：
输入、思考和最终回答共享窗口，要给生成留空间。xhigh 是思考级别，不是固定 token 预算。
表单修改上下文时会同步显式 KV 容量；高级 JSON 编辑器可另设自动 KV。

数据目录的模板 `config/chat_template.jinja` 支持会话中途的 system/developer 消息，避免
“System message must be at the beginning” 错误；扫描不会用模型内置模板替换它。
`config/device-profiles.json` 保留这张 70-SM RTX 5070 Ti 原先的校准，单独 CUDA 13
Native 重校准没有整体改善。它不是通用 GPU 配置；换显卡应使用独立文件校准和验证。
引擎 auto 遇到不匹配设备时仍可能更新该文件。

`NINFER_PREFILL_ALIGN=0` 保留实际 chunk，并清除继承的
`NINFER_PROMPT_FAST`、`CUDA_LAUNCH_BLOCKING` 覆盖。设备 profile 已开启快速
prompt kernel；报告 `fast_prefill_kernel=false` 仅表示没有额外强制开启 CLI 开关。

## 4. 显存策略怎么选

日常只选择一个 `--cuda-memory-policy`。这里“共享系统内存”指显存不足时 Windows
驱动借用系统 RAM 承接 CUDA 设备数据，和主动保存上下文的 CPU 缓存是两回事。

| 参数值 | 界面名称 | 通俗含义 |
|---|---|---|
| `default` | 默认策略 | 按 CUDA 报告可用量安排容量，不额外做严格检查；不保证只用独立显存。 |
| `mixed` | 允许借用系统内存 | 允许实际尝试超过 CUDA 余量的分配，接受 Shared 增长；可能容纳更多数据，但也可能明显变慢。 |
| `strict` | 仅使用独立显存 | 检查设备数据的驻留情况；默认验证 64 MiB 余量，以 128 MiB 步长探测容量。 |
| `strict-128-64` | 自定义严格模式 | 验证 128 MiB 余量，探测步长 64 MiB。两个数字都是 MiB。 |

`strict` 等于 `strict-64-128`。网页选严格模式后才展开余量和步长。余量允许为
0，步长为 1–16384 MiB。严格模式自动选择所需的 Hybrid 上下文缓存，
`--host-cache-mib` 仍独立控制 CPU 缓存大小。

### mixed 不会无限扩容

mixed 表示允许借用系统内存，不是强制把数据放进 Shared。放在哪里仍由驱动决定：

- 显式 KV 容量按请求值尝试分配，失败则报错。
- `--kv-capacity auto` 上限是每个并发请求一个 `--max-context` 窗口，按 KV 页
  向上取整；不继续扩大闲置前缀缓存，也不申请到系统内存耗尽。
- mixed 不使用严格模式的余量和步长。`--kv-headroom-mib` 只用于 default 的自动
  容量；strict 自动容量使用策略里第一个数字作为余量。

mixed 和 strict 当前适用于 Windows 单 GPU 纯文本推理，不能和
`--wddm-evictable-budget` 同用。strict 也不能强制选原缓存或禁用前缀复用。

### strict 检查了什么？

strict 允许受控尝试真实 CUDA 分配，不把 CUDA free 当作唯一硬上限。临时探针释放后，
正式权重和 KV 池仍连续分配，**没有把最终 KV 拆成许多小块**。余量在启动时临时申请并
释放，不会永久占住。

CPU 缓存触碰、传输预热后建立 Shared 基线，再检查 Dedicated/Shared、GPU 可访问性，
以及权重、KV、Graph 的联合布局。“Shared 增长为 0”不是进程 Shared 绝对值为 0：
主动配置的 CPU 缓存可以计入基线。

探测 OOM 可回退，自动 KV 候选有受限退档，但不悄悄缩小显式请求的上下文。Shared 超
基线会拒绝候选；运行中驻留检查失败后，health 和新请求返回 503。它不能永久锁住
Windows/WDDM 的物理放置。一次探针曾比最初 CUDA free 多验证出 640 MiB，下一块因
Shared 增长被拒绝；这不是每次额外获得 640 MiB 的承诺。

任务管理器、NVML/`nvidia-smi`、CUDA free 的口径及采样时刻不同。整卡显示空闲，
不代表 CUDA 此刻一定能申请相同大小的连续设备内存。Graph 观测量是准备区间的
Dedicated 增量，可能包括惰性分配，不是逐笔 Graph 账本。

## 5. 监控页面能看到什么

管理器自身提供监控，不需要 `monitor.py`。每 2 秒采样，当前运行最多保留 6 小时
内存历史；刷新或关闭网页不会清空，退出管理器或开始新一轮模型运行后重置。

| 指标 | 用来判断什么 |
|---|---|
| Prefill、decode、TTFT | 读入提示词和生成的速度，第一次输出要等多久 |
| MTP 接受率、缓存命中与复用路径 | 草稿有多少被接受，旧上下文是否复用 |
| KV 占用、缓存搬运、调度和压力事件 | 上下文占用、搬运是否成为瓶颈 |
| GPU 利用率、功耗、温度、显存 | 显卡是否忙、是否有容量压力 |
| 成功、错误、拒绝请求数 | 本轮累计结果，不拿最近列表长度当总数 |

运行中吞吐按相邻采样的 token 增量计算，已完成请求按实际耗时计算，页面分别标明。
延迟、吞吐、MTP、命中率另有最近 1 小时汇总窗口。日志只增量读取有界尾部，补读大
日志时提示追赶状态。驱动不支持的计数显示无数据，不填零。页面分开显示 NVML 整卡
余量、CUDA 余量、进程 Shared 基线和增长，不把它们混成一个“可分配容量”。

## 6. 当前运行包性能实测：2026-09-29

本节分别记录先前按用户要求停止的主矩阵首轮，以及末尾后来单独完成的 XXS 160K chunk 640/1024 三轮补测；主矩阵没有恢复为三轮测试。两项测试都使用当前运行包里的引擎和模型及测试当时的 AppData 生效配置、模板，没有为测试重新编译引擎。补测完成后，XXS 160K 的默认 chunk 已改为 1024；历史测试表保留各次实际使用的 chunk。

**用户于 9 月 29 日 19:26 要求停止原主矩阵；下方主矩阵各表是首轮实测，不是“三次取最好”。**四种 strict 配置完成了首轮；mixed 完成了 1K、8K、32K、61K 请求，接近 196K 的请求在预填充期间中止。第二、第三轮未运行。共保留 22 个完整正式请求，以及 5 个不计成绩的预热请求。

**测试环境：**RTX 5070 Ti 16 GB（NVML 报告总量 16303 MiB）、Ryzen 7 9800X3D、约 32 GB 系统内存；Windows 11 build 26200、NVIDIA 驱动 617.14、CUDA 13.4.2 / Native SM120a Release，D3D12 residency 关闭。

### 测试方法与指标含义

- 当时共测五种配置：S 64K 和默认 128K 使用 chunk 256；XXS 64K 和当时默认的 160K 使用 chunk 640；额外测试 S mixed 196K、chunk 640。上下文与固定 KV 容量相同，196K 就是 200704 tokens。
- 其他推理参数与第 3 节保持一致：单并发、rk8v4、MTP 2 + ngram 31、graph allowance 72 MiB、Host cache 6144 MiB、开启 xhigh 思考、关闭 lm-head-draft。mixed 显式选择与 strict 相同的备用上下文缓存实现；strict 会自动选择。
- 原计划每种配置独立启动三轮；实际完成的首轮统一使用 seed 42，每种配置都先发小请求预热（输入 93、输出 64 tokens），预热不计成绩。完整正式请求均输出 512 tokens。原定 seed 142、242 的两轮随队列一起停止，没有执行。
- 输入为合成英文长文：重复的实验室观察记录、三个检索键和末尾分析指令。请求开头使用不同前缀避免复用。有效成绩必须是零缓存、root 复用路径、SSE 完整结束，并且客户端和引擎日志的 token 数一致。这是速度与容量测试，没有评价真实编程能力或答案质量。
- 每个有成绩的行都是一次完整请求；“1/3”表示原计划三次、实际测得一次。没有把单次成绩写成三次平均或三次最佳，也不据此宣称小幅差异有统计显著性。
- 预填充速度 = 未缓存输入 tokens / 引擎 prompt 时间；解码速度 =（输出 tokens − 1）/ 引擎 predicted 时间。客户端 TTFT 从发出请求算到第一个非空思考、正文或工具事件，包含预填充。已确认 22/22 个有效请求（100.0%）仅有 512 个思考 tokens，没有最终正文或工具调用。22/22 个请求有思考计数，其中思考占输出 tokens 的 100.0%。TTFT 不是等到最终答案的时间。
- 显卡余量使用**本次请求期间整卡 NVML 可用显存的采样最小值**，单位 MiB，不等于 cudaMalloc 保证可分配的容量。GPU 约每 0.5 秒采样，进程 WDDM 约每秒采样，更短的峰值可能未捕获。

[全部 22 个完整请求数据](assets/rtx5070ti-benchmark-20260929-all.csv)。CSV 还保留服务端 TTFT、总耗时、采样功耗、开启驻留检查时的 CUDA 余量快照、WDDM Dedicated/Shared，以及 MTP/ngram 合计接受率。

### 相同输入对比：62,439 tokens，再生成 512 tokens

| 配置 | chunk | 有效次数 | 预填充 tok/s | 解码 tok/s | 客户端 TTFT 秒 | 总耗时 秒 | 显卡余量 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| S 64K | 256 | 1/3 | 1,671.4 | 111.0 | 37.44 | 42.05 | 1,910 |
| XXS 64K | 640 | 1/3 | 1,692.4 | 106.9 | 36.98 | 41.77 | 3,134 |
| S 128K | 256 | 1/3 | 1,670.2 | 111.0 | 37.47 | 42.08 | 176 |
| XXS 160K | 640 | 1/3 | 1,684.7 | 106.9 | 37.15 | 41.93 | 534 |
| S mixed 196K | 640 | 1/3 | 53.0 | 4.6 | 1,177.16 | 1,288.76 | 24 |

以上为首轮观察值。中止的近 196K 请求没有完整 TTFT 或解码成绩，下方单独记录。

### 不同输入长度的速度

上下文上限不等于这次实际输入长度。请看实际输入列，并给思考和回答留空间。“接近容量”的行输入长度不同，不能当作相同工作量横比。

| 配置 | 实际输入 tokens | 有效次数 | 测量轮次 | 预填充 tok/s | 解码 tok/s | 客户端 TTFT 秒 | 显卡余量 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| S 64K | 993 | 1/3 | 1 | 1,940.1 | 110.4 | 0.53 | 1,910 |
| S 64K | 8,156 | 1/3 | 1 | 2,048.8 | 121.8 | 3.99 | 1,910 |
| S 64K | 32,742 | 1/3 | 1 | 1,863.5 | 123.5 | 17.64 | 1,910 |
| S 64K | 62,439 | 1/3 | 1 | 1,671.4 | 111.0 | 37.44 | 1,910 |
| XXS 64K | 993 | 1/3 | 1 | 1,873.4 | 121.3 | 0.55 | 3,134 |
| XXS 64K | 8,156 | 1/3 | 1 | 1,987.0 | 130.2 | 4.13 | 3,134 |
| XXS 64K | 32,742 | 1/3 | 1 | 1,850.9 | 121.5 | 17.73 | 3,134 |
| XXS 64K | 62,439 | 1/3 | 1 | 1,692.4 | 106.9 | 36.98 | 3,134 |
| S 128K | 993 | 1/3 | 1 | 1,941.0 | 110.2 | 0.51 | 176 |
| S 128K | 8,156 | 1/3 | 1 | 2,041.3 | 121.5 | 4.00 | 176 |
| S 128K | 32,742 | 1/3 | 1 | 1,862.1 | 123.3 | 17.62 | 176 |
| S 128K | 62,439 | 1/3 | 1 | 1,670.2 | 111.0 | 37.47 | 176 |
| S 128K（接近容量） | 127,970 | 1/3 | 1 | 1,358.8 | 92.9 | 94.34 | 176 |
| XXS 160K | 993 | 1/3 | 1 | 1,868.9 | 121.3 | 0.55 | 534 |
| XXS 160K | 8,156 | 1/3 | 1 | 1,985.3 | 130.1 | 4.12 | 534 |
| XXS 160K | 32,742 | 1/3 | 1 | 1,844.3 | 121.8 | 17.79 | 534 |
| XXS 160K | 62,439 | 1/3 | 1 | 1,684.7 | 106.9 | 37.15 | 534 |
| XXS 160K（接近容量） | 160,726 | 1/3 | 1 | 1,317.4 | 93.2 | 122.22 | 534 |
| S mixed 196K | 993 | 1/3 | 1 | 45.2 | 4.9 | 21.98 | 24 |
| S mixed 196K | 8,156 | 1/3 | 1 | 53.0 | 5.2 | 153.87 | 24 |
| S mixed 196K | 32,742 | 1/3 | 1 | 53.4 | 5.2 | 613.58 | 24 |
| S mixed 196K | 62,439 | 1/3 | 1 | 53.0 | 4.6 | 1,177.16 | 24 |
| S mixed 196K（接近容量） | — | 0/3 | — | — | — | — | — |

### 中止的长请求与后续队列

近 196K 请求和后续队列按用户要求停止，不是引擎崩溃或显存分配失败。这轮没有获得接近 196K 输入的完整性能结果。

中止时已耗时约 770.2 秒（12.8 分钟），日志完整区间累计计算约 40960 个输入 tokens，仍在预填充，没有完整 TTFT 和解码速度。计数排除了跨请求边界的日志区间；其中出现的 18 个 decode tokens 属于上一个请求，未计入本次。

- s-mixed-196k / near: 0/3; 没有完整测量。

### 显存与日常配置建议

- **S 64K：**所有有效单次中，整卡采样最低余量为 1,910 MiB；进程 Shared 采样峰值为 6,538 MiB。
- **XXS 64K：**所有有效单次中，整卡采样最低余量为 3,134 MiB；进程 Shared 采样峰值为 6,538 MiB。
- **S 128K：**所有有效单次中，整卡采样最低余量为 176 MiB；进程 Shared 采样峰值为 6,538 MiB。
- **XXS 160K：**所有有效单次中，整卡采样最低余量为 534 MiB；进程 Shared 采样峰值为 6,538 MiB。
- **S mixed 196K：**所有有效单次中，整卡采样最低余量为 24 MiB；进程 Shared 采样峰值为 8,284 MiB。

有效 strict 请求记录的 Shared 基线为 6,538.0 MiB；相对各自基线的采样增量最大为 0.0 MiB（共 18 个请求，基线/增量分别有 18/18 个实测值）。基线包含主动分配的 CPU 上下文缓存，所以 Shared 非零不等于设备数据溢出。mixed 未开启严格驻留检查，CUDA 可用余量一栏留空，不把关闭检查时的零值当成实测。mixed 的 Shared 增长本身也不能单独区分驱动溢出与主动 Host cache。

这张 16 GB 卡上，64K 给桌面和其他 GPU 应用留出的空间明显更多。出厂 S 128K、XXS 160K 更偏向上下文容量，其中 S 128K 的采样余量尤其小。这些是本机验证配置，不是任意 5070 Ti 桌面环境的保证。把上限开大不会自动加速短请求，因为固定 KV 池仍按设定容量分配。

本机不建议日常 agent 使用这组 S mixed 196K / chunk 640：8K 输入整次约 251 秒，61K 输入整次约 21.5 分钟。这里比较的是容量、chunk 和显存策略合在一起的配置，不代表所有 mixed 配置都会有同样的降速。

<!-- xxs-160k-chunks-20260929:start -->
### XXS 160K chunk 640/1024：三轮补测

这是后来单独完成的补测：每个 chunk 独立启动三轮，共六次启动、十二个正式请求。每次先用输入 93、输出 64 tokens 预热（不计成绩），再测 8K 和接近 160K 输入，各生成 512 tokens。上下文和固定 KV 容量仍为 163840，strict 及其他保存参数不变；同轮配对 seed 为 42/142/242。第 1、3 轮顺序为 640→1024，第 2 轮反向。同轮使用完全相同的输入消息，有效启动参数除 chunk 和日志位置外一致。它不是此前已停止主矩阵的续跑。

**日常怎么选：**本机 chunk 1024 在 160K、strict 下独立启动 3 轮都成功。接近 160K 输入的配对结果中，预填充中位提高 1.54%，TTFT 缩短 1.845 秒。代价是采样 GPU 占用多 374 MiB，最低余量从 534 变成 160 MiB。**补测完成后，XXS 160K 默认已改为 chunk 1024**，采用这组长输入表现；需要为其他 GPU 程序留出更多显存时，可以手动改为 chunk 640。

**三次有效请求中的完整最佳单次：**每组按总 wall time 最短选中整次请求。表中每行所有指标来自同一次请求，没有分别挑各列最优值。

| chunk / 长度 | 输入 / 输出 tokens | 轮次 / seed | Prefill tok/s | Decode tok/s | 客户端 / 服务端 TTFT 秒 | 总耗时 秒 | NVML 余量 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 640 / 8K | 8,156 / 512 | 2 / 142 | 1,985.97 | 129.08 | 4.131 / 4.115 | 8.091 | 534 |
| 1024 / 8K | 8,156 / 512 | 2 / 142 | 1,966.74 | 131.60 | 4.170 / 4.155 | 8.055 | 160 |
| 640 / 接近 160K | 160,726 / 512 | 2 / 142 | 1,318.34 | 94.83 | 122.118 / 122.111 | 127.516 | 534 |
| 1024 / 接近 160K | 160,726 / 512 | 3 / 242 | 1,338.62 | 97.97 | 120.291 / 120.263 | 125.517 | 160 |

**1024 相对 640 的同轮配对变化：**列出三对结果的中位数 [最小值, 最大值]，不是把上表两条最佳单次相除。吞吐为正表示更快，TTFT 为负表示缩短。

| 长度 | Prefill 变化 % | Decode 变化 % | 客户端 TTFT 变化 秒 | 客户端 TTFT 变化 % |
|---|---:|---:|---:|---:|
| 8K | -1.01 [-1.01, -0.97] | -3.15 [-10.68, +1.95] | +0.043 [+0.040, +0.046] | +1.03 [+0.96, +1.11] |
| 接近 160K | +1.54 [+1.42, +1.57] | +1.90 [-2.36, +5.51] | -1.845 [-1.867, -1.714] | -1.51 [-1.53, -1.41] |

**显存：**chunk 640/1024 的整卡 NVML 最低采样余量分别为 **534 / 160 MiB**；相对各请求自身基线，strict Shared 增量最大分别为 **0.0 / 0.0 MiB**，记录的基线为 6,538.0 / 6,538.0 MiB。这些是采样口径，不等于保证可申请的空间，也不代表 Windows 驻留被永久锁定；主动 Host cache 会计入 Shared 基线。

每个长度只有三对测量，中位数和范围用于展示这组样本，不宣称统计显著性或普遍提速。六对中有 0/6 对生成内容完全相同；已确认 12/12 个请求只有 512 个思考 tokens，没有最终正文或工具调用。decode 仍受生成文本和草稿接受率影响，TTFT 是首次流输出而不是最终答案时间。测试期间没有修改保存配置；**测试完成后已将 XXS 160K 的保存配置和默认资源同步改为 chunk 1024**，其他启动参数不变。

[十二个补测请求的完整数据](assets/rtx5070ti-xxs-chunks-20260929.csv)还包括服务端 TTFT、采样功耗、CUDA/WDDM 计数、草稿接受率和输出分类，与前面主矩阵的 22 请求 CSV 分开保存。
<!-- xxs-160k-chunks-20260929:end -->

## 7. 构建与适用范围

### 运行条件

已验证 **Windows x64 + RTX 5070 Ti + 驱动 617.14**。这份引擎只保留 SM120a 机器码，
不是适用于所有 CUDA 显卡的通用包。其他计算能力 12.0 显卡预期架构兼容，但没有逐型号
验证；RTX 30/40 系需要相应架构的构建。

CUDA 13.4 的部署建议是 R615 或更新驱动，本包未验证较旧驱动的兼容下限。参考
[NVIDIA GPU 列表](https://developer.nvidia.com/cuda/gpus)、
[Blackwell 兼容说明](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)、
[CUDA 驱动要求](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#cuda-driver)。
依赖齐全时运行无需完整 CUDA Toolkit、编译器、Python、Node.js 或全局 .NET。
换显卡后重验上下文、Graph allowance、余量、校准；这些参数调整通常不用重新编译。
5070 Ti 数据不能直接当作 5090 或 3090 的结果。

### 7.1 从完整源码开始

使用 [Ryan-gsq/ninfer-all 的 `rtx5070ti-cuda13-native` 分支](https://github.com/Ryan-gsq/ninfer-all/tree/rtx5070ti-cuda13-native)，
其中包含本指南、1.3.1 管理器和 strict/mixed 显存策略。上游默认分支或旧提交未必包含这些改动。
首次获取源码时，在 PowerShell 中执行下面的命令；目标目录应尚不存在：

```powershell
git clone --branch rtx5070ti-cuda13-native --single-branch https://github.com/Ryan-gsq/ninfer-all.git C:\src\ninfer-all
```

如果已经有这条分支的完整源码，可以沿用已有目录并调整后面的路径。
以下构建步骤以 `C:\src\ninfer-all` 为例，会先检查关键文件是否存在。

准备这些开发工具；完成构建和转换后，它们不需要放入日常运行目录：

| 工具 | 本路线要求 |
|---|---|
| Git、PowerShell | 下载依赖和执行下面的命令 |
| Visual Studio 2022 Build Tools | “使用 C++ 的桌面开发”、x64 MSVC v143 **14.44.35207**、Windows SDK、C++ CMake/Ninja 工具 |
| CUDA Toolkit | **13.4.2**，包含 nvcc 13.4.92、nvprune 和 cuBLAS 开发/运行文件 |
| CMake / Ninja | 本机 CMake **4.4.3**，新环境建议同版本；项目声明最低 3.28 不等于旧版已验证 CUDA 13.4/120a。Ninja 在 PATH 中 |
| Node.js | **22.12 或更新版本**，与当前锁文件中的 Vite 要求一致 |
| .NET SDK | **10.x**，用于构建 Windows x64 自包含管理器 |
| Python | 3.11，仅转换模型时需要；CPU PyTorch、NumPy 和下载工具按转换教程安装 |

CUDA 从 [13.4.2 官方归档](https://developer.nvidia.com/cuda-13-4-2-download-archive)
安装。下面假设 VS Build Tools 和 CUDA 使用常见安装位置；若不同，修改路径。不要只把
`cl.exe` 加到 PATH：C++ 头文件、库和 Windows SDK 也需要开发环境初始化。

在**同一个 PowerShell 窗口**依次执行后面的步骤。`$packageRoot` 是一个新的输出目录，
先不要覆盖正在使用的 Downloads 目录。四并发是本机 32 GB RAM 构建时的选择，可按内存
调整 `$jobs`。

```powershell
$ErrorActionPreference = 'Stop'
$ninferRepo = 'C:\src\ninfer-all'
$buildRoot = Join-Path $ninferRepo 'build-5070ti-native'
$vcpkgRoot = 'C:\src\vcpkg'
$cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
$publishRoot = Join-Path $ninferRepo 'dist\windows-manager'
$packageRoot = 'C:\build\ninfer-package\qwen27b'
$jobs = 4

Set-Location $ninferRepo
foreach ($required in @(
  'apps/windows-manager/NInfer.Manager.csproj',
  'apps/windows-manager/config/profiles/xxs-160k.json',
  'src/product/cuda_memory_options.h',
  'tools/convert/__main__.py'
)) {
  if (-not (Test-Path -LiteralPath (Join-Path $ninferRepo $required))) {
    throw "Incomplete source checkout: $required"
  }
}

$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" -vcvars_ver=14.44 >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] }
}
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment setup failed' }
$clPath = (Get-Command cl.exe -ErrorAction Stop).Source
$env:CUDA_PATH = $cudaRoot
$env:PATH = "$cudaRoot\bin\x64;$cudaRoot\bin;$env:PATH"
(Get-Item -LiteralPath $clPath).VersionInfo.FileVersion
& "$cudaRoot\bin\nvcc.exe" --version
cmake --version
ninja --version
```

确认输出对应 MSVC 14.44（编译器文件版本通常显示 19.44）和 nvcc 13.4.92。
这里不使用旧 3090 配方的构建树；已有正确配置的 5070 Ti 构建目录应直接增量构建。

### 7.2 安装依赖并构建 CUDA 算子

这一条标准路线使用仓库根目录的 `vcpkg.json`：它固定 baseline 并声明 curl、
FFmpeg（含 zlib）和 pkgconf。CMake 首次配置通过 vcpkg 工具链下载和构建依赖，
可能花较长时间；不需要手工拼接 FFmpeg include/lib。
[Microsoft 的 manifest 集成说明](https://learn.microsoft.com/en-us/vcpkg/users/buildsystems/cmake-integration)
说明了这项自动安装行为。

下面用于新构建目录，不要给已有的“预构建依赖桥接”目录换工具链。初次配置成功后，
依赖位于 `$buildRoot\vcpkg_installed\x64-windows`。

```powershell
if (-not (Test-Path -LiteralPath $vcpkgRoot)) {
  git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
  if ($LASTEXITCODE -ne 0) { throw 'vcpkg clone failed' }
}
& "$vcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
if ($LASTEXITCODE -ne 0) { throw 'vcpkg bootstrap failed' }

cmake -S $ninferRepo -B $buildRoot -G Ninja `
  "-DCMAKE_TOOLCHAIN_FILE=$vcpkgRoot/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows -DVCPKG_MANIFEST_MODE=ON `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a `
  -DNINFER_SM120_NATIVE=ON -DNINFER_BUILD_APPS=ON `
  -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF `
  -DNINFER_DIRECTSTORAGE=OFF -DNINFER_D3D12_RESIDENCY=OFF `
  "-DCUDAToolkit_ROOT=$cudaRoot" `
  "-DCMAKE_CUDA_COMPILER=$cudaRoot/bin/nvcc.exe" `
  "-DCMAKE_CUDA_HOST_COMPILER=$clPath" `
  "-DCMAKE_C_COMPILER=$clPath" "-DCMAKE_CXX_COMPILER=$clPath"
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
cmake --build $buildRoot --target ninfer_ops -j $jobs
if ($LASTEXITCODE -ne 0) { throw 'CUDA operators build failed' }
```

配置采用 Release、`120a`、Native ON、D3D12 residency OFF、DirectStorage OFF。
Native 路线启用 PDL，Windows 使用 staged TMA descriptors；独立兼容路线选项为 OFF
不代表这些 Native 路径关闭。没有加入未经本机验证的额外 cuBLAS/A8 性能开关。

### 7.3 裁剪 archive 后链接引擎

本机的 CUDA 算子 archive 同时带有冗余 PTX，直接链接可能超过 Windows PE 的 2 GiB
上限。先备份，再用工具包自己的 `nvprune.exe` 只保留 SM120a SASS，成功后替换链接
输入；最终仍是正常 Release 构建。此步骤不适用于多架构发行包。

```powershell
$opsArchive = Join-Path $buildRoot 'src\ops\ninfer_ops.lib'
$prunedArchive = Join-Path $buildRoot 'src\ops\ninfer_ops.native.lib'
$backupRoot = Join-Path $buildRoot 'archive-backup'
New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
$backupName = 'ninfer_ops-' + (Get-Date -Format 'yyyyMMdd-HHmmssfff') + '.with-ptx.lib'
Copy-Item -LiteralPath $opsArchive -Destination (Join-Path $backupRoot $backupName)

& "$cudaRoot\bin\nvprune.exe" -arch sm_120a $opsArchive -o $prunedArchive
if ($LASTEXITCODE -ne 0) { throw 'nvprune failed; original archive was not replaced' }
Copy-Item -LiteralPath $prunedArchive -Destination $opsArchive -Force

cmake --build $buildRoot --target ninfer-serve -j $jobs
if ($LASTEXITCODE -ne 0) { throw 'Engine link failed' }
```

产物在 `$buildRoot\apps\ninfer-serve.exe`。以后若修改 CUDA 算子，先构建
`ninfer_ops`、重新裁剪，再链接；不要在 archive 重建后跳过裁剪。可执行文件仍会很大，
因为包含大量架构专用 CUDA kernel，文件大小不能单独判断 Debug/Release。
这些端到端检查也不代表全部 NVFP4/MoE 数学路线已经完成数值验证。

### 7.4 转换 GSQ/RCO 模型

按 [下载与 GSQ GGUF 转换教程](rtx-5070ti-windows-downloads.md) 准备 Swift 或 ISTA
GGUF 和对应配置/分词器，使用同一份源码的 `tools.convert`。该教程包含 Python 环境、
下载文件、`qwen3_8_27b_gguf`、`text,mtp`、CPU 转换和 `--proposal` 的完整命令。

若要得到本页两个默认配置，分别转换 Swift `IQ3_XXS` 和 `IQ3_S`，输出保持教程中的
文件名并放在仓库 `converted-models/`。原始 GGUF、元数据和转换报告是制作材料，
不放入最终日常包；转换器也不随运行包发布。转换不会编译 CUDA 引擎，也不需要再次
量化已经选定的 GGUF。

### 7.5 构建网页和托盘管理器

先按锁文件安装 npm 依赖并构建网页，再 publish C#。Vite 输出到
`apps/windows-manager/wwwroot/`，C# 项目会把它复制到发布目录。先后顺序不能颠倒，
否则可能发布旧网页或缺少网页。

```powershell
node --version
dotnet --list-sdks
Push-Location (Join-Path $ninferRepo 'apps\windows-manager\web')
try {
  npm.cmd ci
  if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
  npm.cmd run build
  if ($LASTEXITCODE -ne 0) { throw 'Website build failed' }
} finally { Pop-Location }

dotnet publish (Join-Path $ninferRepo 'apps\windows-manager\NInfer.Manager.csproj') `
  -c Release -r win-x64 --self-contained true `
  -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true `
  -o $publishRoot
if ($LASTEXITCODE -ne 0) { throw 'Manager publish failed' }
```

`$publishRoot` 中的 EXE 已捆绑 .NET runtime。模型转换用的 Python、前端构建用的
Node.js 和 .NET SDK 都不需要随包提供。仅修改管理器时也不用重新编译 CUDA。

### 7.6 组装一个新的运行目录

下面复制完整的 vcpkg **Release bin** DLL 集合，包含 FFmpeg/curl 的传递依赖，再添加
同一 CUDA 工具包的 cuBLAS/Lt 和 x64 VC runtime。不能只复制 CMake apps 目录里刚好
出现的几个 DLL；也不要混用另一套 FFmpeg/CUDA 的同名文件。Windows 路线静态链接
cudart，但这里随同工具包运行库一起携带，以覆盖依赖需求。VC runtime 从 MSVC 授权的
Redist 目录复制，相关安装要求见
[Microsoft VC Redistributable 说明](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)。

**仓库默认 profile 和运行目录 profile 的外层格式不同。** 仓库
`apps/windows-manager/config/profiles/*.json` 是内嵌种子，顶层直接是 `id`、
`modelPath` 等；磁盘生效文件需要 `schemaVersion / savedAt / profile` 外层。
因此下面为它们生成封装，不能直接递归复制整个源码 config 目录当作运行配置。
另一种有效做法是不提供包内 config，让管理器首次启动完全从内嵌种子初始化。

```powershell
if (Test-Path -LiteralPath $packageRoot) {
  throw 'Choose a new, empty packageRoot; do not overwrite a running installation'
}
foreach ($directory in @('', 'engine', 'model', 'config\profiles', 'docs\assets', 'licenses')) {
  New-Item -ItemType Directory -Path (Join-Path $packageRoot $directory) -Force | Out-Null
}
Copy-Item -LiteralPath (Join-Path $publishRoot 'NInferManager.exe') -Destination $packageRoot
Copy-Item -LiteralPath (Join-Path $publishRoot 'wwwroot') -Destination $packageRoot -Recurse
$engineDir = Join-Path $packageRoot 'engine'
Copy-Item -LiteralPath (Join-Path $buildRoot 'apps\ninfer-serve.exe') -Destination $engineDir

$dependencyRoot = Join-Path $buildRoot 'vcpkg_installed\x64-windows'
Get-ChildItem -LiteralPath (Join-Path $dependencyRoot 'bin') -File -Filter '*.dll' |
  Copy-Item -Destination $engineDir

foreach ($dll in @('cublas64_13.dll', 'cublasLt64_13.dll', 'cudart64_13.dll')) {
  $source = Join-Path $cudaRoot "bin\$dll"
  if (-not (Test-Path -LiteralPath $source)) { $source = Join-Path $cudaRoot "bin\x64\$dll" }
  if (-not (Test-Path -LiteralPath $source)) { throw "Missing CUDA DLL: $dll" }
  Copy-Item -LiteralPath $source -Destination $engineDir
}
$crtRoot = Join-Path $env:VCToolsRedistDir 'x64\Microsoft.VC143.CRT'
if (-not (Test-Path -LiteralPath $crtRoot)) {
  throw 'Locate the installed MSVC x64 redistributable directory and set crtRoot'
}
Get-ChildItem -LiteralPath $crtRoot -File -Filter '*.dll' | Copy-Item -Destination $engineDir

$seedRoot = Join-Path $ninferRepo 'apps\windows-manager\config'
Get-ChildItem -LiteralPath $seedRoot -File | Copy-Item -Destination (Join-Path $packageRoot 'config')
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $seedRoot 'profiles') -File -Filter '*.json') {
  $profile = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
  $document = [ordered]@{
    schemaVersion = 1
    savedAt = [DateTimeOffset]::UtcNow.ToString('O')
    profile = $profile
  }
  $target = Join-Path $packageRoot "config\profiles\$($file.Name)"
  [IO.File]::WriteAllText($target, ($document | ConvertTo-Json -Depth 32), [Text.UTF8Encoding]::new($false))
}

foreach ($name in @('rtx-5070ti-windows.md', 'rtx-5070ti-windows.en.md', 'rtx-5070ti-windows-downloads.md')) {
  Copy-Item -LiteralPath (Join-Path $ninferRepo "docs\$name") -Destination (Join-Path $packageRoot 'docs')
}
foreach ($name in @('ninfer-tray-running.png', 'ninfer-tray-stopped.png', 'rtx5070ti-benchmark-20260929-all.csv', 'rtx5070ti-xxs-chunks-20260929.csv')) {
  Copy-Item -LiteralPath (Join-Path $ninferRepo "docs\assets\$name") -Destination (Join-Path $packageRoot 'docs\assets')
}
Copy-Item -LiteralPath (Join-Path $ninferRepo 'LICENSE') -Destination $packageRoot
foreach ($port in Get-ChildItem -LiteralPath (Join-Path $dependencyRoot 'share') -Directory) {
  $copyright = Join-Path $port.FullName 'copyright'
  if (Test-Path -LiteralPath $copyright) {
    $licenseDir = Join-Path $packageRoot "licenses\$($port.Name)"
    New-Item -ItemType Directory -Path $licenseDir -Force | Out-Null
    Copy-Item -LiteralPath $copyright -Destination $licenseDir
  }
}
Get-ChildItem -LiteralPath $cudaRoot -File |
  Where-Object { $_.Name -match 'LICENSE|EULA' } |
  Copy-Item -Destination (Join-Path $packageRoot 'licenses')
```

接着放入已转换的模型。只需要 XXS 时可从 `$entryNames` 中去掉 S 那一行；
如果改了文件名或使用 ISTA，后续在网页中选择实际模型文件，保存自己的 profile。
下面按各自 `.conversion.json` 报告中的 `files` 列表复制主文件和全部分卷，不猜测卷名，
也不会覆盖已有模型文件。报告与转换输出先保留在原转换位置，复制完成后不必放入运行包。

```powershell
$convertedModelRoot = Join-Path $ninferRepo 'converted-models'
$entryNames = @(
  'Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.ninfer',
  'Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.ninfer'
)
$modelDir = Join-Path $packageRoot 'model'
$filesToCopy = @(foreach ($entryName in $entryNames) {
  $reportPath = Join-Path $convertedModelRoot ($entryName + '.conversion.json')
  $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if (-not $report.files) { throw "Conversion report has no files: $reportPath" }
  foreach ($file in $report.files) {
    [pscustomobject]@{
      source = $file.path
      target = Join-Path $modelDir ([IO.Path]::GetFileName($file.path))
    }
  }
})
foreach ($file in $filesToCopy) {
  if (-not (Test-Path -LiteralPath $file.source -PathType Leaf)) { throw "Missing model volume: $($file.source)" }
  if (Test-Path -LiteralPath $file.target) { throw "Model file already exists: $($file.target)" }
}
foreach ($file in $filesToCopy) {
  [IO.File]::Copy($file.source, $file.target, $false)
}
```

这会形成第 2 节的 PackageRoot 结构：一个管理器、一个引擎目录、模型、网页、
首次配置种子和文档。没有 converter、构建缓存、源码测试脚本或另一套旧引擎。
转发软件时保留项目、CUDA、VC runtime 和所带依赖的适用许可文件。

### 7.7 独立目录检查与首次启动

先临时去掉开发工具的 PATH，再执行引擎 `--help`，确认它不会偷偷依赖开发目录里的
DLL。这只检查基本装载，不代表已验证所有推理路径。若报 0xC0000135，通常是某个直接或
传递 DLL 缺失；检查 `engine/`，不要靠保留开发环境 PATH 掩盖它。

随后用 `--no-autostart --open` 只打开管理器，检查模型配置后再手动启动。

```powershell
$savedPath = $env:PATH
try {
  $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
  & (Join-Path $packageRoot 'engine\ninfer-serve.exe') --help
  if ($LASTEXITCODE -ne 0) { throw 'Packaged engine failed to start; inspect runtime dependencies' }
} finally { $env:PATH = $savedPath }

Start-Process -FilePath (Join-Path $packageRoot 'NInferManager.exe') `
  -ArgumentList @('--root', "`"$packageRoot`"", '--no-autostart', '--open') `
  -WindowStyle Hidden
```

管理器会在这个安装位置对应的 AppData 中初始化配置。网页保存后的参数在 DataRoot，
不是包内种子；已有 DataRoot 时修改包内 config 不会覆盖它。模板和设备 profile 从
DataRoot 读取，engine/model 从 PackageRoot 读取。把检查通过的整个目录移到最终位置
（例如 Downloads/qwen27b）后会生成另一个安装位置 ID，按第 2 节重新初始化。

从网页确认模型就绪、发送一个短请求、观察日志和显存，然后停止。停止模型后管理器
应仍可打开；退出管理器才一起结束监控。注册登录自启需要手动开启，不因这条测试命令
自动开启。

**复现审计的范围：**本节核对了仓库 CMake、vcpkg manifest、实际成功构建脚本、管理器
资源格式和已经完成的部署验证。本机成功引擎使用 MSVC 14.44、CUDA 13.4.2 和预构建
FFmpeg/curl 桥接；本次没有在空白 Windows 上重新安装完整 vcpkg/CUDA/VS 并重跑全部
构建，也没有把不同依赖版本下的结果冒充为逐位复现。以上标准 vcpkg 路线仍需要读者在
自己的开发机上完成依赖构建和基本启动检查。

### 7.8 已完成的管理器验证

管理器 1.3.1 的 Release 构建完成，0 warning、0 error；117 项后端检查、37 项平台
检查通过，后者包含 19 项隔离自启检查。使用真实 Windows ACL 拒绝程序目录写入时，
仍可正常使用 AppData 保存数据；该测试结束后已恢复 ACL。

实际部署也已验证：18 个配置文件内容不变地导入 AppData，XXS 160K 成功启动，模板和
设备 profile 参数指向 AppData 的 `config/`，日志写入 AppData 的 `logs/`。API 短请求
返回 `OK`（输入 16、输出 2 tokens），随后停止模型。**尚未真正重启并重新登录**，隔离
自启检查不能替代这一验证；这项管理器启动验证与第 6 节性能测量分别记录。

管理器检查无需 GPU，从仓库根目录执行：

```powershell
dotnet build tests/windows-manager-platform/PlatformTest.csproj -c Release
dotnet tests/windows-manager-platform/bin/Release/net10.0-windows/PlatformTest.dll
dotnet run --project tests/windows-manager-backend/BackendCheck.csproj -c Release
```

平台检查用 `dotnet <assembly>`，让 fake child 和 signal helper 共享 host。可选
`NINFER_TEST_ARTIFACT` 只做真实模型头部检查，不读取权重或初始化 GPU。
