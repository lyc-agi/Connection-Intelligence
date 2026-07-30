from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from .connection import Connection


class ContradictionType(Enum):
    """矛盾类型。"""
    MISMATCH = "mismatch"               # 连接度不匹配（期望 vs 实际）
    CONSTRAINT_VIOLATION = "constraint_violation"  # 约束违反
    CIRCULAR_DEPENDENCY = "circular_dependency"    # 循环依赖
    CONFLICTING_DEMANDS = "conflicting_demands"   # 不同连接的需求冲突
    EXTERNAL_PRESSURE = "external_pressure"        # 外部压力导致的矛盾


class Contradiction:
    """
    矛盾 - 连接度之间的不匹配。

    根据智能理论，矛盾的本质在于连接程度的不匹配。
    这种不匹配可能存在于：
    - 期望连接度与实际连接度之间
    - 多个连接的约束之间
    - 事物的内部属性与外部连接之间

    智能的作用就是检测并解决这些矛盾。
    """

    def __init__(
        self,
        contradiction_type: ContradictionType,
        description: str,
        involved_connections: Optional[List[Connection]] = None,
        severity: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.contradiction_type = contradiction_type
        self.description = description
        self.involved_connections: List[Connection] = involved_connections or []
        self.severity = max(0.0, min(1.0, severity))
        self.context: Dict[str, Any] = context or {}
        self.resolved = False
        self.resolution_history: List[Dict[str, Any]] = []

    @property
    def connection_ids(self) -> List[str]:
        return [c.id for c in self.involved_connections]

    @property
    def total_strain(self) -> float:
        """
        计算矛盾涉及的所有连接的总应变。
        """
        return sum(c.strain() for c in self.involved_connections)

    def detect(self) -> bool:
        """
        检测矛盾是否仍然存在。
        基于涉及的连接的当前状态。
        """
        if self.resolved:
            return False

        if self.contradiction_type == ContradictionType.MISMATCH:
            for conn in self.involved_connections:
                if conn.mismatch > 0.01:
                    return True
            return False

        elif self.contradiction_type == ContradictionType.CONSTRAINT_VIOLATION:
            for conn in self.involved_connections:
                if not conn.satisfies_constraints():
                    return True
            return False

        elif self.contradiction_type == ContradictionType.CONFLICTING_DEMANDS:
            if len(self.involved_connections) >= 2:
                c1, c2 = self.involved_connections[0], self.involved_connections[1]
                if c1.mismatch > 0.1 and c2.mismatch > 0.1:
                    return True
            return False

        elif self.contradiction_type == ContradictionType.CIRCULAR_DEPENDENCY:
            return self.context.get('has_cycle', False)

        elif self.contradiction_type == ContradictionType.EXTERNAL_PRESSURE:
            return self.severity > 0.05

        return False

    def mark_resolved(self, method: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.resolved = True
        self.resolution_history.append({
            'method': method,
            'details': details or {},
            'severity_at_resolution': self.severity,
        })

    def adjust_severity(self, new_severity: float) -> None:
        self.severity = max(0.0, min(1.0, new_severity))

    def __repr__(self) -> str:
        status = "RESOLVED" if self.resolved else "ACTIVE"
        return (f"<Contradiction [{status}]: {self.contradiction_type.value} "
                f"severity={self.severity:.3f} - {self.description[:50]}>")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Contradiction):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
