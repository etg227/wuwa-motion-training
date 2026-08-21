# Character Motion Cycle Dataset

> [!NOTE]
> **拆库建议**：`training/` 已增长到约 7000 行，依赖（torch / mss / pynput /
> tkinter）与启动器完全独立，也不在 `deploy.txt` 分发清单和 CI 覆盖范围内，
> 建议迁入独立仓库。目录布局（`training/motion/` + `training_data/`）设计为
> 可整体平移——在新仓库根目录保持同样的相对路径即可，`common.py` 的
> `ROOT`/`DATA_ROOT` 与 `auto_train.py` 的子进程调用无需任何修改。迁移命令：
>
> ```powershell
> # 在 GitHub 上新建空仓库 <owner>/wuwa-motion-training 后（保留原路径深度与历史）：
> git clone https://github.com/<owner>/wuwa-yg-launcher.git motion-split
> cd motion-split
> py -3.12 -m pip install git-filter-repo
> git filter-repo --path training/ --path tests/TestPhaseTracker.py --path tests/TestReadyEvidence.py
> git remote add origin https://github.com/<owner>/wuwa-motion-training.git
> git push -u origin main
> # 确认新仓库可用后，再在本仓库删除 training/ 与上述两个测试文件。
> ```
>
> 注意保留 `training/motion/` 的目录深度（不要把 `motion/` 提升到仓库根），
> 否则 `common.py` 的 `ROOT = parents[2]` 会指向仓库外。

这个实验训练集用于从连续游戏视频学习角色的**整套平A循环相位**，而不是预先规定角色有 3 段、4 段或 5 段平A。

核心假设非常简单：连续平A时，只要某个可辨认动作/姿态再次出现，就说明完整平A循环经过了一轮。相邻两个“相同姿态”边界之间就是一个 `cycle`。

模型最终输出 `0.0 ~ 1.0` 的循环相位：

```text
同一参考姿态        -> 0%
中间所有动作        -> 1% ... 99%
再次回到参考姿态    -> 下一轮 0%
```

因此**不需要填写平A段数，也不需要把每次鼠标点击当成 A1/A2/A3 标签**。玩家持续连点 A 或 E 也不会直接污染动作标签；后续可以把输入日志作为 READY/ACCEPTED 的辅助数据单独使用。

## 数据目录

真正的视频、cycle、模型都写到仓库外观上的本地目录 `training_data/`，并已加入 `.gitignore`：

```text
training_data/motion/
  Suisui/
    videos/
    annotations/
    cycles/
    models/
    manifest.jsonl
  Chisa/
  Jiyan/
  ...
```

每个角色一个目录，但**没有 combo_stages 配置**。

## 第一次准备环境

主程序环境不强制安装 PyTorch。训练环境单独安装：

```powershell
py -3.12 -m pip install -r training/motion/requirements.txt
```

有 NVIDIA CUDA 环境时请按 PyTorch 官方对应 CUDA 版本安装 `torch/torchvision`；本脚本会自动优先使用 CUDA，没有则使用 CPU。

## 明天最简单的采集方式

先从一个角色做 PoC，例如穗穗。录 20~60 秒视频，在同一段里**一直平A**即可。为了泛化，建议后续逐渐补充：不同地图、敌人、镜头角度、命中/未命中、不同画质和 FPS。

社区/B站视频也可以作为本地素材导入，`source=community` 只是记录来源类型；工具本身不负责下载视频。使用公开视频训练前请自行确认素材使用许可和平台/作者要求。

导入你自己的录屏：

```powershell
py -3.12 training/motion/import_video.py --character Suisui --video "D:\video\suisui_basic.mp4" --source self
```

导入已经保存到本地的攻略视频：

```powershell
py -3.12 training/motion/import_video.py --character Suisui --video "D:\video\guide.mp4" --source community
```

导入命令会把视频复制到：

```text
training_data/motion/Suisui/videos/
```

## 标记“同一个动作再次出现”

不需要标 A1/A2/A3。打开连续平A视频：

```powershell
py -3.12 training/motion/mark_cycles.py --character Suisui --video "training_data\motion\Suisui\videos\<video>.mp4"
```

操作：

```text
SPACE   当前帧标记为 cycle boundary
G       已有前两个相同姿态后，自动建议后续 boundary
X       撤销最后一个标记
A / D   前后 1 帧
J / L   前后 10 帧
P       播放 / 暂停
S       保存
Q       保存并退出
```

建议先选一个每轮都比较好辨认的姿态，在第一次出现时 `SPACE`，下一轮同一姿态再次出现时再 `SPACE`。然后按 `G`：工具会使用这两个点估算周期，并在后续预期位置附近用角色区域视觉相似度寻找最像参考姿态的帧。

自动建议不是最终真值；请播放/逐帧检查明显错位的边界。相邻两个 boundary 就是一整套平A，无论里面实际有几段。

默认 ROI 是画面中央大区域。如果人物位置特殊可以手动指定 normalized ROI：

```powershell
--roi 0.10,0.05,0.80,0.90
```

## 构建训练 cycle

标好多个视频后：

```powershell
py -3.12 training/motion/build_dataset.py --character Suisui
```

它会把每两个 boundary 之间的视频按原始时间顺序抽成一个 cycle，保存连续 RGB 帧以及每帧对应的 `phase=0..1`。

默认：

```text
30 FPS
112 x 112 ROI
每个 cycle 保持自己的真实长度
```

这一步不会把不同长度的动作强行压成同一个固定时长。

## 训练

```powershell
py -3.12 training/motion/train_phase_model.py --character Suisui --epochs 30
```

第一版模型是轻量 CNN + GRU：每次看最近 12 帧，然后输出一个二维圆周向量，转换成 `0..1` 的 combo phase。

使用 `sin/cos` 圆周目标是为了让：

```text
phase 99%
```

和：

```text
phase 1%
```

在数学上仍然彼此接近，因为它们本来就是同一个动作循环边界附近。

模型保存在：

```text
training_data/motion/Suisui/models/phase_model.pt
```

最开始至少需要 3 个 cycle 才能跑；实际建议先准备 **30+ 个完整 cycle**，并尽量来自多个录屏。不要只录同一个背景几十遍。

## 验证模型有没有真的学会一整套平A

```powershell
py -3.12 training/motion/infer_phase_video.py ^
  --video "D:\video\suisui_test.mp4" ^
  --model "training_data\motion\Suisui\models\phase_model.pt"
```

画面顶部会显示 `combo phase`。如果模型正确，连续平A过程中相位应大致持续前进，并在同一套动作重新开始时从接近 100% 回到 0%。终端也会打印 `cycle wrap`。

## 为什么不拿连续点击直接当标签

玩家真实操作往往是：

```text
A A A A A A ...
E E E E ...
```

很多输入只是“提前请求”，游戏并没有接受。把每一次点击都标成新动作会产生错误监督。因此当前 phase dataset 只相信**画面循环**。

以后第二阶段再加入输入 telemetry：

```text
视觉：A动作正在运行
输入：玩家已经开始 spam A
视觉：下一动作真正开始
```

由此反推出 `CHAIN_READY / SKILL_READY / ACCEPTED` 窗口。也就是说：

```text
视频 phase 模型负责：角色现在运动到哪里
输入 burst 数据负责：什么时候开始尝试下一招最合适
```

两者分开训练。

## 下一步目标

第一阶段只验证一个问题：**不告诉模型平A段数，只靠连续视频，它能否稳定跟踪完整平A循环相位。**

如果穗穗 PoC 成功，再把同样流程用于 Chisa、YangYangSp、Jiyan 等角色。每个角色都可以拥有完全不同的动作长度和段数，训练管线不需要修改。
