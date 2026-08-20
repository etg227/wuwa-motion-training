# 一键角色动作训练

这一版不再要求人工截图、数平A段数或手工标 cycle。角色名第一次确认后，日常采集只做两件事：

```text
开始录制
→ 回到鸣潮一直平A / 正常操作
→ 回来点击“停止录制并自动训练”
```

启动：

```powershell
py -3.12 training/motion/auto_train.py
```

## 自动完成的工作

点击“开始录制”后，工具先找到 `client-win64-shipping.exe`。只有鸣潮真正处于前台时才写入视频；Alt+Tab 或回来点击停止时自动暂停，因此训练器窗口不会混进录像。

录制期间同时保存键盘/鼠标 down/up 与对应视频帧号。这些输入**不会**被当成 A1/A2/A3 标签；它们只是为后续学习 `READY / ACCEPTED` 窗口保留原始 telemetry，所以连续狂点 A/E 不会直接污染当前 phase 模型。

点击停止后自动执行：

```text
录像保存
→ 从角色 ROI 生成外观 + 边缘运动特征
→ 在连续视频中搜索稳定重复周期
→ 自动选择每轮都重复出现的参考姿态
→ 自动生成完整 cycle boundaries
→ 低置信度录像拒绝进入训练集
→ 重建该角色全部 cycle dataset
→ CUDA 可用时自动用 GPU 训练 CNN + GRU phase model
```

整个流程不知道角色到底有 3、4 还是 5 段平A。它只学习一整套动作从 `0% → 100% → 下一轮 0%` 的循环相位。

## 第一次使用

安装独立训练依赖：

```powershell
py -3.12 -m pip install -r training/motion/requirements.txt
```

运行：

```powershell
py -3.12 training/motion/auto_train.py
```

第一次在“角色”中填例如 `Suisui`。它会被记住；之后同一个角色每次采集只需要点击开始和停止。

为了让自动周期发现更稳，单次建议至少录 20~60 秒，并尽量让画面中包含多轮连续完整平A。后续可以换地图、敌人、镜头和命中条件继续录；每次停止都会重新使用该角色此前积累的全部有效 cycle 训练。

## 自动质量保护

自动周期发现不会为了“必须训练”而强行制造标签。如果录像太短、动作重复不稳定或相似度置信度太低：

- 原始 MP4 与输入 telemetry 仍然保存；
- 本段不会写入 `.cycles.json`；
- 已有训练数据和模型不会被污染；
- 界面日志会说明为什么本次没有进入训练。

## 本地数据

所有训练素材仍然位于被 `.gitignore` 排除的：

```text
training_data/motion/<Character>/
  videos/
    auto_*.mp4
    auto_*.inputs.jsonl
    auto_*.session.json
  annotations/
  cycles/
  models/phase_model.pt
  manifest.jsonl
```

录像和模型不会被提交到 GitHub。
