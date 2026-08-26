# UP 视频轴提取

目标是把高手 / UP 主视频里的**成功操作结果与时序**提取成可分析、可编译的轴，而不是要求本地玩家先把同一套高难轴复现出来。

数据职责分开：

```text
本机 video + telemetry      → 只教“这个视觉变化像 A / E / Q / R / 切人”
UP 主视频                    → 提供真正的高手动作顺序、严格切人位置与循环 timing
UP 视频自身重复循环           → 无监督学习 recurring visual event / phase
```

因此本机手柄操作上限不会成为高手轴 timing 的教师上限。

## 推荐：全自动第一遍

```powershell
py -3.12 training/axis/auto_extract.py `
  --video "C:\path\to\up.mp4" `
  --start-slot 1
```

默认会按源视频帧率分析事件（最高 60fps），另外降采样到约 6fps 只用于寻找 8–45 秒的重复队伍循环；所以 30fps 源仍保留约 33.3ms 的原始视觉时间粒度。

它会自动做：

1. 扫描主体、右侧队伍 HUD、右下技能 HUD 的视觉变化；
2. 自动找动作边界候选并聚成 recurring visual clusters；
3. 自动寻找重复 team rotation，并把不同循环中相同 phase 的事件互相对齐；
4. 如果本机已有 `training_data/motion/**/auto_*.mp4 + *.inputs.jsonl`，自动把 telemetry 当弱监督，只学习视觉语义原型；
5. 高置信度事件才写进可编译 timeline；不能确定的事件保持 unknown，不强行猜成 A/E/Q/R；
6. 只把重要、低置信度的少量片段放入 review 清单。

默认还会在兄弟目录 `../wuwa-yg-launcher/training_data/motion/` 寻找拆库前留下的 telemetry 数据。也可以显式指定：

```powershell
py -3.12 training/axis/auto_extract.py `
  --video "up.mp4" `
  --prototype-root "D:\github\wuwa-motion-training\training_data\motion"
```

如果完全不想使用自己的录像：

```powershell
py -3.12 training/axis/auto_extract.py --video "up.mp4" --no-self-prototypes
```

这种模式仍会自动学习重复循环、视觉事件 cluster 和切人候选，但不知道的语义会保持 unknown。

## 自动输出

假设输入是 `up.mp4`：

```text
up.auto_analysis.json        完整自动分析：事件、cluster、HUD 变化、置信度、语义候选
up.loops.json                重复循环边界 + recurring event phase 模板
up.review.json               只列优先复核的小片段
up.auto_axis_timeline.json   仅高置信度且能映射到 a/e/q/r/s1/s2/s3 的事件
up.analysis.txt              简短摘要
```

`auto_axis_timeline.json` 的存在**不代表已经可以直接上号跑**。它故意只收录高置信度语义事件；严格切人和 unknown recurring cluster 应先看 `auto_analysis.json / review.json`，后续再转成 phase/cancel window。

### 30fps 的意义

30fps 源每帧约 33.3ms。工具不会通过插帧假装恢复不存在的真实输入帧。

它仍然可以可靠学习：

- 动作顺序；
- 大部分切人视觉边界；
- 重复循环结构；
- “某个高手切人发生在 outgoing action 的哪一段 / 哪个 recurring phase”。

后续严格切人应优先转成**视觉 phase / cancel window**，而不是声称知道 UP 主精确到 5ms 的物理按键时间。

## 人工标注器：只作为修正工具

自动提取无法判断的少量片段，才使用：

```powershell
py -3.12 training/axis/annotate_video.py --video up.mp4 --start-slot 1
```

| 键 | 含义 |
| --- | --- |
| `, / .` | 上一帧 / 下一帧 |
| `J / L` | -10 / +10 帧 |
| `P` | 播放 / 暂停 |
| `A` | 普攻（左键） |
| `V` | 下落 A |
| `E / Q / R / F / W` | 对应技能键 |
| `Z` | 重击（长按左键） |
| `I` | 变奏入场（对齐标记，不产生输入） |
| `1 / 2 / 3` | 切人，同时更新当前站场 |
| `X` | 撤销最后一个事件 |
| `S` | 保存 |
| `Esc` | 保存并退出 |

标注器中的 `ms` 始终表示**动作在画面上出现**的时刻，不要人工预估按键提前量。

## 编译人工/确认后的 timeline

```powershell
py -3.12 training/axis/compile_axis.py --timeline up.axis_timeline.json
```

视觉→输入 lead 的优先级：

```text
transition lead > action lead > global lead
```

例如：

```powershell
py -3.12 training/axis/compile_axis.py `
  --timeline up.axis_timeline.json `
  --action-lead "a=45,e=80,r=110,s2=60" `
  --transition-lead "e:s2=95,a:s3=70"
```

这些 lead 只用于把**视觉出现时刻**转换成估计输入时刻；高手轴的相对 timing 仍来自 UP 视频本身，而不是要求本机玩家复现同一难度。

## 当前自动提取边界

- 无监督 recurring cluster 能发现“这里每轮都发生同类视觉事件”，但没有足够 telemetry / HUD 证据时不会强行命名为 E/Q/R；
- 右侧队伍 HUD 的大变化会标成 swap candidate；没有可靠证据区分目标槽位时会进 review，不会乱编 `s1/s2/s3`；
- 当前 prototype 是轻量视觉弱监督，不是已经训练好的跨角色 ActionNet；后续可把自动积累的高置信度事件继续训练成模型；
- 一次只支持一个按住中的输入，不支持“按住 W 同时按 E”这类和弦宏编译；
- 视频请自行保存为本地文件；本仓库不负责平台下载与分发。
