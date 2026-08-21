# Wuwa Motion Training

鸣潮角色动作循环的视频训练、相位识别与离线验证实验仓库。

本仓库从 [`etg227/wuwa-yg-launcher`](https://github.com/etg227/wuwa-yg-launcher) 的动作训练实验分支拆出，并保留相关 Git 历史。启动器本体、角色战斗逻辑与固定队伍轴仍在原仓库维护；这里仅维护训练与实验验证工具。

## 当前范围

- 连续游戏视频与键鼠 / XInput telemetry 采集；
- 自动发现重复动作 cycle，并构建 phase dataset；
- 多动作模式聚类、phase 模型训练与跨录像相位对齐；
- 基于真人输入 telemetry 的 ATTACK / CHAIN_READY pseudo-supervised 学习；
- 离线 replay、phase tracker、READY evidence 等验证工具；
- 受控 live probe 实验代码。

> `live_ready_probe.py` 与 `pydirect_transport_probe.py` 会产生真实游戏输入，属于实验代码。普通训练和离线验证不需要运行它们。

## 目录

```text
training/motion/       训练、推理与验证代码
tests/                 训练模块相关测试
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

## 本地数据

训练素材和生成模型默认位于：

```text
training_data/motion/<Character>/
```

这些视频、telemetry、cycle、模型和 replay 不进入 Git。拆库不会自动搬运原 `wuwa-yg-launcher/training_data/` 中的本地数据；如果需要继续使用旧数据，请在本机自行复制到本仓库相同路径。

## 测试

当前拆出的训练专属测试为：

```text
tests/TestPhaseTracker.py
tests/TestReadyEvidence.py
tests/TestLiveReadyProbe.py
```

可在仓库根目录使用 Python `unittest` 运行。部分训练 / 录像流程需要 Windows、OpenCV、PyTorch 或游戏环境，因此不应把“能导入测试模块”等同于完整实机验证。

## 许可证与来源

训练代码来自 `wuwa-yg-launcher` 的 AGPL-3.0 代码历史，本仓库继续按 GNU Affero General Public License v3.0 使用（见 `LICENSE.txt`）。原项目基础来自 OK-WW / OK-Script；具体来源与致谢以 `wuwa-yg-launcher` 项目说明为准。
