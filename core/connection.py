from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional


class ConnectionType(Enum):
    """连接类型枚举。"""
    CAUSAL = "causal"           # 因果连接
    SPATIAL = "spatial"         # 空间连接
    TEMPORAL = "temporal"       # 时间连接
    FUNCTIONAL = "functional"   # 功能连接
    HIERARCHICAL = "hierarchical"  # 层次连接
    SIMILARITY = "similarity"   # 相似连接
    DEPENDENCY = "dependency"   # 依赖连接
    CUSTOM = "custom"           # 自定义连接


class Connection:
    """
    连接 - 事物之间的关系。

    连接的核心属性是"度"(degree)，表示两个事物之间的连接强度：
    - 0 表示完全没有连接
    - 1 表示最强连接
    - 度是智能可以调节的核心参数

    多种具体的连接形式可以被抽象为统一的连接度。
    """

    def __init__(
        self,
        source_id: str,
        target_id: str,
        degree: float = 0.5,
        connection_type: ConnectionType = ConnectionType.CUSTOM,
        weight: float = 1.0,
        expected_degree: Optional[float] = None,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.source_id = source_id
        self.target_id = target_id
        self._degree = 0.5
        self.connection_type = connection_type
        self.weight = weight
        self._expected_degree: Optional[float] = None
        self._constraints: list[dict] = []

        self.degree = degree
        if expected_degree is not None:
            self.expected_degree = expected_degree

    @property
    def degree(self) -> float:
        return self._degree

    @degree.setter
    def degree(self, value: float) -> None:
        self._degree = max(0.0, min(1.0, value))

    @property
    def expected_degree(self) -> Optional[float]:
        return self._expected_degree

    @expected_degree.setter
    def expected_degree(self, value: float) -> None:
        self._expected_degree = max(0.0, min(1.0, value))

    @property
    def mismatch(self) -> float:
        """
        连接度的不匹配程度。
        当实际度与期望度不一致时，产生不匹配 (0-1)。
        这是矛盾的来源。
        """
        if self._expected_degree is None:
            return 0.0
        return abs(self._degree - self._expected_degree)

    def add_constraint(self, constraint: dict) -> None:
        """
        添加连接约束。

        约束示例:
        {"type": "range", "min": 0.2, "max": 0.8}
        {"type": "depends_on", "connection_id": "...", "delta": 0.1}
        """
        self._constraints.append(constraint)

    @property
    def constraints(self) -> list[dict]:
        return self._constraints.copy()

    def satisfies_constraints(self) -> bool:
        for c in self._constraints:
            if c['type'] == 'range':
                if not (c.get('min', 0.0) <= self._degree <= c.get('max', 1.0)):
                    return False
        return True

    def strain(self) -> float:
        """
        计算连接的应变 (strain)。
        应变越大，说明连接的当前状态与期望/约束的冲突越大。
        """
        s = self.mismatch * self.weight
        for c in self._constraints:
            if c['type'] == 'range':
                lo = c.get('min', 0.0)
                hi = c.get('max', 1.0)
                if self._degree < lo:
                    s += (lo - self._degree) * self.weight
                elif self._degree > hi:
                    s += (self._degree - hi) * self.weight
        return s

    def relax(self, learning_rate: float = 0.1) -> float:
        """
        松弛连接度以减小应变。返回调整量。
        """
        old_degree = self._degree
        if self._expected_degree is not None:
            target = self._expected_degree
        else:
            target = self._degree

        for c in self._constraints:
            if c['type'] == 'range':
                lo = c.get('min', 0.0)
                hi = c.get('max', 1.0)
                target = max(lo, min(hi, target))

        delta = (target - self._degree) * learning_rate
        self.degree += delta
        return abs(self._degree - old_degree)

    def is_between(self, thing_a_id: str, thing_b_id: str) -> bool:
        return (self.source_id == thing_a_id and self.target_id == thing_b_id) or \
               (self.source_id == thing_b_id and self.target_id == thing_a_id)

    def __repr__(self) -> str:
        return (f"<Connection: {self.source_id} -> {self.target_id} "
                f"degree={self._degree:.3f} type={self.connection_type.value}>")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connection):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
