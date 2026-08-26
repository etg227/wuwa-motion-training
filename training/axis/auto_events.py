"""UP 视频自动轴提取的纯 NumPy 事件签名与输出护栏。

把“结构发现”和“语义分类”使用的视觉时间尺度明确拆开：

- local_event_signatures：只看事件前后相邻 analysis sample，服务 visual cluster / recurrence；
- semantic_event_signatures：使用 -60ms 到 +40..220ms 的长视界，服务 telemetry prototype 分类；
- semantic_timeline_guard：语义标签明显退化时 fail closed，禁止生成可编译 timeline。

本模块不依赖 OpenCV，便于轻量 CI 覆盖关键安全逻辑。
"""

from __future__ import annotations

from collections import Counter

import numpy as np


DEFAULT_SEMANTIC_BEFORE_MS = 60.0
DEFAULT_SEMANTIC_AFTER_OFFSETS_MS = (40.0, 70.0, 100.0, 135.0, 175.0, 220.0)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if norm < 1e-6:
        return np.zeros_like(value, dtype=np.float32)
    return value / norm


def _dimension(body: np.ndarray, party: np.ndarray, ability: np.ndarray) -> int:
    return int(body.shape[1] + party.shape[1] + ability.shape[1])


def _signature(
    before: tuple[np.ndarray, np.ndarray, np.ndarray],
    after: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    parts = [np.abs(after[index] - before[index]) for index in range(3)]
    return _normalize_vector(np.concatenate(parts))


def local_event_signatures(
    body: np.ndarray,
    party: np.ndarray,
    ability: np.ndarray,
    indexes: list[int],
) -> np.ndarray:
    """构造 visual clustering 用的短视界签名。

    每个候选只比较前后 1 个 analysis sample。这个表示刻意保持局部，避免
    semantic classifier 为匹配 telemetry 而扩大时间视界后，反过来改变
    visual cluster / recurrence 的结构定义。
    """

    count = len(body)
    dimension = _dimension(body, party, ability)
    rows: list[np.ndarray] = []

    for raw_index in indexes:
        index = int(raw_index)
        before_index = max(0, index - 1)
        after_index = min(count - 1, index + 1)
        before = (
            body[before_index],
            party[before_index],
            ability[before_index],
        )
        after = (
            body[after_index],
            party[after_index],
            ability[after_index],
        )
        rows.append(_signature(before, after))

    if not rows:
        return np.empty((0, dimension), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def semantic_event_signatures(
    body: np.ndarray,
    party: np.ndarray,
    ability: np.ndarray,
    indexes: list[int],
    analysis_fps: float,
    *,
    before_ms: float = DEFAULT_SEMANTIC_BEFORE_MS,
    after_offsets_ms: tuple[float, ...] = DEFAULT_SEMANTIC_AFTER_OFFSETS_MS,
) -> np.ndarray:
    """构造 telemetry prototype 分类用的长视界签名。

    与 prototype 构建保持一致：before 取事件前约 60ms；after 在 40..220ms
    候选中选择原始变化量最大的一帧。这里只服务动作语义，不参与 visual cluster。
    """

    if analysis_fps <= 0:
        raise ValueError("analysis_fps must be > 0")

    count = len(body)
    dimension = _dimension(body, party, ability)
    before_offset = max(1, int(round(float(before_ms) * analysis_fps / 1000.0)))
    after_offsets = sorted({
        max(1, int(round(float(offset) * analysis_fps / 1000.0)))
        for offset in after_offsets_ms
    })

    rows: list[np.ndarray] = []
    for raw_index in indexes:
        index = int(raw_index)
        before_index = max(0, index - before_offset)
        before = (
            body[before_index],
            party[before_index],
            ability[before_index],
        )

        best_signature = None
        best_change = -1.0
        for offset in after_offsets:
            after_index = min(count - 1, index + offset)
            if after_index <= before_index:
                continue
            after = (
                body[after_index],
                party[after_index],
                ability[after_index],
            )
            raw_change = sum(
                float(np.linalg.norm(after[part] - before[part]))
                for part in range(3)
            )
            if raw_change > best_change:
                best_change = raw_change
                best_signature = _signature(before, after)

        if best_signature is None:
            best_signature = np.zeros(dimension, dtype=np.float32)
        rows.append(best_signature)

    if not rows:
        return np.empty((0, dimension), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def semantic_timeline_guard(
    labels: list[str],
    *,
    min_events: int = 8,
    max_dominant_share: float = 0.85,
) -> str | None:
    """检测语义标签退化；命中时返回阻断原因，否则返回 None。

    当足够多的语义事件里绝大多数都被判成同一 token，通常意味着 prototype
    失衡或分类器退化，而不是高手轴真的只有一种动作。命中后调用方应保留
    analysis/review/loops，但把可编译 timeline 写空。
    """

    cleaned = [str(label) for label in labels if label]
    if len(cleaned) < max(1, int(min_events)):
        return None

    token, count = Counter(cleaned).most_common(1)[0]
    share = count / len(cleaned)
    if share <= float(max_dominant_share):
        return None

    return (
        f"标签退化：{len(cleaned)} 个语义事件里 {share:.0%} 都是 '{token}'。"
        "已阻止生成可编译 timeline；请补齐语义原型或降低错误分类后重跑。"
    )
