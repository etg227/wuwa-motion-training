# Wuwa Motion Training

鸣潮角色动作循环的视频训练、相位识别、高手视频轴提取与离线验证实验仓库。

本仓库从 [`etg227/wuwa-yg-launcher`](https://github.com/etg227/wuwa-yg-launcher) 的动作训练实验分支拆出，并保留相关 Git 历史。启动器本体、角色战斗逻辑与固定队伍轴仍在原仓库维护；这里仅维护训练与实验验证工具。

## 当前范围

- 连续游戏视频与键鼠 / XInput telemetry 采集；
- 自动发现重复动作 cycle，并构建 phase dataset；
- 多动作模式聚类、phase 模型训练与跨录像相位对齐；
- 基于真人输入 telemetry 的 ATTACK / CHAIN_READY pseudo-supervised 学习；
- 从高手 / UP 视频自动发现动作边界、重复队伍循环、严格切人候选与 recurring phase；
- 将高置信度视频事件整理为轴 timeline，人工逐帧标注仅作为低置信度修正工具；
- 离线 replay、phase tracker、READY evidence 等验证工具；
- 受控 live probe 实验代码。

> `live_ready_probe.py` 与 `pydirect_transport_probe.py` 会产生真实游戏输入，属于实验代码。普通训练、UP 视频分析和离线验证不需要运行它们。

## 目录

```text
training/motion/       训练、推理与验证代码
training/axis/         UP 视频自动轴提取、低置信度复核与 timeline 编译
tests/                 训练与轴提取模块的测试
training_data/motion/  本地视频、telemetry、cycle、模型与 replay（Git 忽略）
```

保留 `training/motion/` 这个目录深度是有意为之：`common.py` 使用 `Path(__file__).resolve().parents[2]` 计算仓库根目录，因此训练数据会稳定写入仓库根的 `training_data/motion/`。

## 环境

需要 Python 3.12。建议使用独立虚拟环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.12 -m pip install -r training/motion/requirements.txt
```

## 最简单的自动采集 / 训练

```powershell
py -3.12 training/motion/auto_train.py
```

选择角色后点击“开始录制”，回到游戏连续执行目标动作；结束时点击“停止录制并自动训练”。工具会保存视频与输入 telemetry、自动寻找 cycle、重建数据集并训练 phase 模型。详细说明见 [`training/motion/AUTO_TRAIN.md`](training/motion/AUTO_TRAIN.md)。

更完整的手工导入、cycle 标记、phase 训练和验证说明见 [`training/motion/README.md`](training/motion/README.md)。

## 自动分析高手 / UP 视频

不要求本机先复现同一条高难轴。自己的 telemetry 只作为视觉语义弱监督，真正的动作顺序、循环 timing 和严格切人位置来自高手视频本身。

```powershell
py -3.12 training/axis/auto_extract.py --video "C:\path\to\up.mp4" --start-slot 1
```

工具会输出完整分析、重复循环、少量低置信度 review 清单，以及只包含高置信度语义事件的 `auto_axis_timeline.json`。详细说明见 [`training/axis/README.md`](training/axis/README.md)。

## 本地数据

训练素材和生成模型默认位于：

```text
training_data/motion/<Character>/
```

这些视频、telemetry、cycle、模型和 replay 不进入 Git。拆库不会自动搬运原 `wuwa-yg-launcher/training_data/` 中的本地数据；如果需要继续使用旧数据，请在本机自行复制到本仓库相同路径。

## 测试

```powershell
py -3.12 -m unittest discover tests -p "Test*.py" -v
```

CI 只安装 NumPy 并覆盖纯逻辑单元测试；部分训练、OpenCV 视频分析、录像与 live probe 流程需要额外依赖、Windows 或游戏环境，因此不应把轻量 CI 通过等同于完整实机验证。

## 许可证与来源

训练代码来自 `wuwa-yg-launcher` 的 AGPL-3.0 代码历史，本仓库继续按 GNU Affero General Public License v3.0 使用（见 `LICENSE.txt`）。原项目基础来自 OK-WW / OK-Script；具体来源与致谢以 `wuwa-yg-launcher` 项目说明为准。
