from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core.thing import Thing
from core.connection import Connection
from core.contradiction import Contradiction


class IntelligenceDegree:
    """
    智能度量器 - 根据理论计算智能程度。

    核心公式:
    D = Q - P

    其中:
    - P: 无智能干预时矛盾解决的概率 (自然解决率)
    - Q: 有智能干预时矛盾解决的概率 (智能解决率)
    - D: 智能程度，取值范围 (-1, 1)

    D > 0 才能视为智能，值越大则智能程度越高。
    """

    def __init__(self, natural_relaxation_rate: float = 0.1):
        self.natural_relaxation_rate = natural_relaxation_rate
        self._history: List[Dict] = []

    def compute(
        self,
        contradictions: List[Contradiction],
        things: Dict[str, Thing],
        connections: List[Connection],
        resolution_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        计算当前的智能程度 D。

        返回:
        {
            'D': 智能程度值,
            'P': 自然解决概率,
            'Q': 智能解决概率,
            'details': 详细分解,
        }
        """
        if not contradictions and not connections:
            return {'D': 0.0, 'P': 0.0, 'Q': 0.0, 'details': {}}

        # 计算当前矛盾的总严重度
        total_severity = sum(c.severity for c in contradictions) if contradictions else 0.0
        unresolved_count = len([c for c in contradictions if not c.resolved])
        resolved_count = len([c for c in contradictions if c.resolved])

        # P: 自然解决概率
        # 基于自然界"弛豫"原理，矛盾有自发缓解的趋势
        # 复杂矛盾自然解决概率低，简单矛盾自然解决概率高
        P = self._estimate_natural_resolution_probability(
            total_severity, unresolved_count, connections
        )

        # Q: 智能解决概率
        Q = self._estimate_intelligent_resolution_probability(
            contradictions, resolution_history
        )

        # 当所有矛盾都被解决时，D 应反映智能的成功
        # 而不是简单地返回 0
        if unresolved_count == 0 and resolved_count > 0:
            # 智能成功解决了所有矛盾
            # P: 自然情况下解决全部矛盾的概率很低
            P_success = self._estimate_natural_resolution_probability(
                total_severity, resolved_count, connections
            )
            # Q: 智能解决全部矛盾的概率很高
            Q_success = 0.9  # 智能确实解决了所有矛盾
            D = Q_success - P_success
        elif unresolved_count == 0 and resolved_count == 0:
            # 没有矛盾，D = 0
            D = 0.0
        else:
            # D = Q - P
            D = Q - P

        result = {
            'D': D,
            'P': P,
            'Q': Q,
            'details': {
                'total_severity': total_severity,
                'unresolved_count': unresolved_count,
                'total_contradictions': len(contradictions),
                'resolved_count': len([c for c in contradictions if c.resolved]),
                'total_things': len(things),
                'total_connections': len(connections),
            }
        }

        self._history.append(result)
        return result

    def _estimate_natural_resolution_probability(
        self,
        total_severity: float,
        unresolved_count: int,
        connections: List[Connection],
    ) -> float:
        """
        估算自然解决概率 P。

        自然解决概率基于:
        1. 矛盾的总严重度 - 越严重越难自然解决
        2. 连接网络的大小 - 网络越大，自然波动越可能解决矛盾
        3. 自然弛豫率 - 基本的自然缓解速度
        """
        if unresolved_count == 0:
            return 1.0

        # 基础自然缓解率
        base_rate = self.natural_relaxation_rate

        # 严重度因子: 严重度越高，自然解决概率越低
        severity_factor = max(0.1, 1.0 - total_severity / max(unresolved_count, 1))

        # 网络大小因子: 更多的连接意味着更多的自然波动可能性
        network_factor = min(1.0, len(connections) / 10.0)

        # 复杂矛盾 (多连接涉及) 更难自然解决
        complexity_penalty = 1.0
        if unresolved_count > 5:
            complexity_penalty = 0.7

        P = base_rate * severity_factor * network_factor * complexity_penalty
        return max(0.0, min(1.0, P))

    def _estimate_intelligent_resolution_probability(
        self,
        contradictions: List[Contradiction],
        resolution_history: Optional[List[Dict]] = None,
    ) -> float:
        """
        估算智能解决概率 Q。

        智能解决概率基于:
        1. 历史解决率 - 智能在过去解决矛盾的成功率
        2. 当前策略的有效性
        3. 矛盾的可解性
        """
        if not contradictions:
            return 1.0

        resolved = [c for c in contradictions if c.resolved]
        unresolved = [c for c in contradictions if not c.resolved]

        if not unresolved:
            return 1.0

        # 历史解决率
        if resolution_history:
            total_attempts = len(resolution_history)
            if total_attempts > 0:
                successful = sum(1 for h in resolution_history
                                if h.get('result', {}).get('success', False))
                historical_rate = successful / total_attempts
            else:
                historical_rate = 0.5
        else:
            historical_rate = 0.5

        # 当前可解性: 基于矛盾类型
        type_bonus = 0.0
        for c in unresolved:
            if c.contradiction_type.value == 'mismatch':
                type_bonus += 0.8  # 不匹配最容易解决
            elif c.contradiction_type.value == 'constraint_violation':
                type_bonus += 0.6
            elif c.contradiction_type.value == 'conflicting_demands':
                type_bonus += 0.4
            elif c.contradiction_type.value == 'circular_dependency':
                type_bonus += 0.5
            else:
                type_bonus += 0.3

        avg_solvability = type_bonus / len(unresolved) if unresolved else 0.5

        # 结合历史和当前可解性
        Q = 0.6 * historical_rate + 0.4 * avg_solvability

        return max(0.0, min(1.0, Q))

    def compute_dynamic(
        self,
        contradiction_before: Contradiction,
        contradiction_after: Contradiction,
    ) -> Dict:
        """
        动态计算单个矛盾的解决对智能度的贡献。

        这衡量的是: 智能解决一个具体矛盾的效率。
        """
        severity_before = contradiction_before.severity
        severity_after = contradiction_after.severity

        # 如果矛盾被解决了
        if contradiction_after.resolved and not contradiction_before.resolved:
            # 智能带来的改善
            improvement = severity_before - severity_after
            # 归一化到 (-1, 1)
            D_single = improvement * 2 - 0.2  # 减去自然解决率
        else:
            D_single = 0.0

        return {
            'D_single': D_single,
            'severity_before': severity_before,
            'severity_after': severity_after,
            'resolved': contradiction_after.resolved,
        }

    @property
    def average_D(self) -> float:
        """
        计算历史平均智能程度。
        """
        if not self._history:
            return 0.0
        return sum(h['D'] for h in self._history) / len(self._history)

    @property
    def best_D(self) -> float:
        """
        历史最高智能程度。
        """
        if not self._history:
            return 0.0
        return max(h['D'] for h in self._history)

    @property
    def history(self) -> List[Dict]:
        return self._history.copy()

    def reset(self) -> None:
        self._history.clear()
