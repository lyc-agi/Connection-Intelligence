from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from core.thing import Thing
from core.connection import Connection, ConnectionType
from core.contradiction import Contradiction, ContradictionType
from core.law import LawLibrary, Law
from core.network import NetworkGraph


class ResolutionLevel(IntEnum):
    """
    矛盾解决的抽象层次。

    6.5 原理: 智能能够跳出矛盾所在的层次，从更高层次解决。
    有些矛盾在当前层次内无法解决，但跳出层次则很容易解决。

    层次从低到高，抽象程度递增:
    - Level 0: 直接层 - 直接调整连接度
    - Level 1: 结构层 - 重组连接网络结构
    - Level 2: 规律层 - 利用规律变换连接
    - Level 3: 目的层 - 重新评估目的和期望
    - Level 4: 妥协层 - 让矛盾互相妥协
    """
    DIRECT = 0       # 直接调整连接度
    STRUCTURAL = 1   # 重组网络结构
    LAW_BASED = 2    # 利用规律
    PURPOSE = 3      # 重新评估目的
    COMPROMISE = 4   # 矛盾妥协


class HierarchicalResolver:
    """
    分层矛盾解决器 - 实现 6.5 原理。

    核心思想:
    1. 先尝试在当前层次解决矛盾
    2. 如果失败，提升到更高抽象层次
    3. 每提升一层，可用的解决手段更多
    4. 最高层次是让矛盾互相妥协

    智能在解决矛盾的同时必然带来了新的矛盾，
    只是新的矛盾在智能看来的影响更小。
    """

    def __init__(
        self,
        law_library: Optional[LawLibrary] = None,
        network: Optional[NetworkGraph] = None,
        learning_rate: float = 0.15,
    ):
        self.law_library = law_library or LawLibrary()
        self.network = network or NetworkGraph()
        self.learning_rate = learning_rate
        self._resolution_log: List[Dict[str, Any]] = []
        self._new_contradictions: List[Contradiction] = []

    def resolve(
        self,
        contradiction: Contradiction,
        things: Dict[str, Thing],
        connections: List[Connection],
        start_level: ResolutionLevel = ResolutionLevel.DIRECT,
    ) -> Dict[str, Any]:
        """
        分层解决矛盾。

        从 start_level 开始，逐级提升直到解决或到达最高层。
        """
        # 构建网络图
        conn_dict = {c.id: c for c in connections}
        self.network.build_from(things, conn_dict)

        for level in range(start_level, len(ResolutionLevel)):
            current_level = ResolutionLevel(level)
            result = self._resolve_at_level(
                contradiction, things, connections, current_level
            )

            if result.get('success', False):
                result['level'] = current_level.value
                result['level_name'] = current_level.name
                result['contradiction_id'] = contradiction.id

                # 记录可能产生的新矛盾
                new_conns = result.get('new_contradictions', [])
                if new_conns:
                    result['new_contradiction_count'] = len(new_conns)
                    result['note'] = (
                        '解决矛盾的同时产生了新的矛盾，'
                        '但新矛盾的影响更小 (6.5原理)'
                    )

                self._resolution_log.append(result)
                return result

        # 所有层次都失败
        result = {
            'success': False,
            'level': ResolutionLevel.COMPROMISE.value,
            'level_name': ResolutionLevel.COMPROMISE.name,
            'contradiction_id': contradiction.id,
            'message': '所有层次均未能完全解决矛盾',
        }
        self._resolution_log.append(result)
        return result

    def _resolve_at_level(
        self,
        contradiction: Contradiction,
        things: Dict[str, Thing],
        connections: List[Connection],
        level: ResolutionLevel,
    ) -> Dict[str, Any]:
        """在特定层次解决矛盾。"""
        if level == ResolutionLevel.DIRECT:
            return self._resolve_direct(contradiction, connections)
        elif level == ResolutionLevel.STRUCTURAL:
            return self._resolve_structural(contradiction, things, connections)
        elif level == ResolutionLevel.LAW_BASED:
            return self._resolve_law_based(contradiction, things, connections)
        elif level == ResolutionLevel.PURPOSE:
            return self._resolve_purpose(contradiction, connections)
        elif level == ResolutionLevel.COMPROMISE:
            return self._resolve_compromise(contradiction, connections)
        return {'success': False}

    # ==================== Level 0: 直接层 ====================

    def _resolve_direct(
        self,
        contradiction: Contradiction,
        connections: List[Connection],
    ) -> Dict[str, Any]:
        """
        Level 0: 直接调整连接度。

        最简单的策略，通过逐步调整实际度向期望度靠拢。
        适用于轻度不匹配。
        """
        adjustments = []
        success = True

        for conn in contradiction.involved_connections:
            old_degree = conn.degree
            old_strain = conn.strain()

            if conn.expected_degree is not None:
                # 沿期望方向调整
                delta = (conn.expected_degree - conn.degree) * self.learning_rate
                conn.degree += delta

                adjustments.append({
                    'connection_id': conn.id,
                    'old_degree': old_degree,
                    'new_degree': conn.degree,
                    'strain_reduction': old_strain - conn.strain(),
                })

                if conn.mismatch > 0.05:
                    success = False  # 一次调整不够
            elif not conn.satisfies_constraints():
                # 约束违反：调整到可行域
                old_degree = conn.degree
                for c in conn.constraints:
                    if c['type'] == 'range':
                        lo, hi = c.get('min', 0), c.get('max', 1)
                        if conn.degree < lo:
                            conn.degree = lo + (lo - conn.degree) * self.learning_rate
                        elif conn.degree > hi:
                            conn.degree = hi - (conn.degree - hi) * self.learning_rate

                adjustments.append({
                    'connection_id': conn.id,
                    'old_degree': old_degree,
                    'new_degree': conn.degree,
                    'action': 'constraint_fix',
                })

                if not conn.satisfies_constraints():
                    success = False

        if adjustments and success:
            contradiction.mark_resolved('direct', {'adjustments': adjustments})

        return {
            'success': success,
            'strategy': 'direct_adjustment',
            'adjustments': adjustments,
            'new_contradictions': [],  # 直接层不产生新矛盾
        }

    # ==================== Level 1: 结构层 ====================

    def _resolve_structural(
        self,
        contradiction: Contradiction,
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> Dict[str, Any]:
        """
        Level 1: 重组网络结构。

        通过移动、重定向或重组连接来解决矛盾。
        包括:
        - 查找桥梁节点建立新路径
        - 移除循环中的最弱连接
        - 重定向连接到替代节点
        """
        new_contradictions = []
        modifications = []

        if contradiction.contradiction_type == ContradictionType.CIRCULAR_DEPENDENCY:
            # 打破循环：移除最弱连接
            cycle = contradiction.context.get('cycle', [])
            if cycle:
                cycle_conns = []
                for i in range(len(cycle) - 1):
                    for conn in contradiction.involved_connections:
                        if conn.source_id == cycle[i] and conn.target_id == cycle[i + 1]:
                            cycle_conns.append(conn)

                if cycle_conns:
                    weakest = min(cycle_conns, key=lambda c: c.weight)
                    old_degree = weakest.degree
                    weakest.degree = 0.1  # 大幅减弱而非完全移除

                    modifications.append({
                        'connection_id': weakest.id,
                        'old_degree': old_degree,
                        'new_degree': weakest.degree,
                        'action': 'break_cycle',
                    })

                    # 新矛盾：被减弱的连接可能产生新的不匹配
                    if weakest.expected_degree and weakest.expected_degree > 0.2:
                        new_contradictions.append(Contradiction(
                            contradiction_type=ContradictionType.MISMATCH,
                            description=f'打破循环后连接 {weakest.id} 度过低',
                            involved_connections=[weakest],
                            severity=abs(weakest.degree - weakest.expected_degree) * 0.3,
                        ))

                    contradiction.mark_resolved('structural', {'modifications': modifications})
                    return {
                        'success': True,
                        'strategy': 'break_cycle',
                        'modifications': modifications,
                        'new_contradictions': [
                            {'id': c.id, 'severity': c.severity} for c in new_contradictions
                        ],
                    }

        # 查找桥梁节点
        for conn in contradiction.involved_connections:
            if conn.mismatch > 0.2:
                bridges = self.network.find_bridge_nodes(conn.source_id, conn.target_id)
                if bridges:
                    # 通过桥梁节点建立间接路径
                    bridge_id = bridges[0]
                    modifications.append({
                        'action': 'use_bridge',
                        'bridge_node': bridge_id,
                        'for_connection': conn.id,
                    })

                    # 增强到桥梁的连接
                    for c in connections:
                        if (c.source_id == conn.source_id and c.target_id == bridge_id) or \
                           (c.source_id == bridge_id and c.target_id == conn.target_id):
                            old_degree = c.degree
                            c.degree = min(1.0, c.degree + 0.2)
                            modifications.append({
                                'connection_id': c.id,
                                'old_degree': old_degree,
                                'new_degree': c.degree,
                                'action': 'strengthen_bridge',
                            })

                    # 减弱原来的直接连接
                    old_degree = conn.degree
                    conn.degree = max(0.1, conn.degree * 0.5)
                    modifications.append({
                        'connection_id': conn.id,
                        'old_degree': old_degree,
                        'new_degree': conn.degree,
                        'action': 'reroute',
                    })

                    contradiction.mark_resolved('structural', {'modifications': modifications})
                    return {
                        'success': True,
                        'strategy': 'reroute_via_bridge',
                        'modifications': modifications,
                        'new_contradictions': [],
                    }

        return {
            'success': False,
            'strategy': 'structural',
            'modifications': modifications,
            'new_contradictions': [],
        }

    # ==================== Level 2: 规律层 ====================

    def _resolve_law_based(
        self,
        contradiction: Contradiction,
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> Dict[str, Any]:
        """
        Level 2: 利用规律解决矛盾。

        6.4 原理: 智能利用规律来变换连接。
        从规律库中选择适用的规律并应用。
        """
        applicable_laws = self.law_library.find_applicable(
            things, connections, contradiction
        )

        if not applicable_laws:
            return {
                'success': False,
                'strategy': 'law_based',
                'message': '没有适用的规律',
            }

        results = []
        for law in applicable_laws:
            result = law.apply(things, connections, contradiction)
            results.append({
                'law_name': law.name,
                'law_type': law.law_type.value,
                'result': result,
            })

            if result.get('success', False):
                contradiction.mark_resolved('law_based', {
                    'law_used': law.name,
                    'result': result,
                })

                # 规律应用可能产生新矛盾
                new_contradictions = []
                for mod in result.get('modifications', []):
                    conn_id = mod.get('connection_id')
                    if conn_id:
                        conn = next((c for c in connections if c.id == conn_id), None)
                        if conn and conn.expected_degree:
                            new_mismatch = conn.mismatch
                            if new_mismatch > 0.1 and new_mismatch < contradiction.severity:
                                new_contradictions.append(Contradiction(
                                    contradiction_type=ContradictionType.MISMATCH,
                                    description=f'规律应用后连接 {conn_id} 产生轻度不匹配',
                                    involved_connections=[conn],
                                    severity=new_mismatch * 0.5,  # 新矛盾影响更小
                                ))

                return {
                    'success': True,
                    'strategy': 'law_based',
                    'law_used': law.name,
                    'law_result': result,
                    'all_applicable_laws': [l.name for l in applicable_laws],
                    'new_contradictions': [
                        {'id': c.id, 'severity': c.severity} for c in new_contradictions
                    ],
                }

        return {
            'success': False,
            'strategy': 'law_based',
            'results': results,
            'message': '所有适用规律均未成功',
        }

    # ==================== Level 3: 目的层 ====================

    def _resolve_purpose(
        self,
        contradiction: Contradiction,
        connections: List[Connection],
    ) -> Dict[str, Any]:
        """
        Level 3: 重新评估目的。

        6.5 原理: 智能选择不同的目的可以使之增强或减弱，或者避开它。
        在这一层，智能不再试图改变现实（连接度），
        而是改变期望（目的），使之与现实更一致。
        """
        adjustments = []

        for conn in contradiction.involved_connections:
            if conn.expected_degree is not None:
                old_expected = conn.expected_degree
                old_mismatch = conn.mismatch

                # 将期望向实际靠拢（但不完全放弃目的）
                # 保留 30% 的原始期望差距
                gap = conn.expected_degree - conn.degree
                conn.expected_degree = conn.degree + gap * 0.3

                adjustments.append({
                    'connection_id': conn.id,
                    'old_expected': old_expected,
                    'new_expected': conn.expected_degree,
                    'old_mismatch': old_mismatch,
                    'new_mismatch': conn.mismatch,
                })

        success = all(
            conn.mismatch < 0.1 for conn in contradiction.involved_connections
            if conn.expected_degree is not None
        ) if adjustments else False

        if success:
            contradiction.mark_resolved('purpose', {'adjustments': adjustments})

        # 目的层产生的新矛盾：期望降低可能影响其他连接
        new_contradictions = []
        for adj in adjustments:
            if adj['new_mismatch'] > 0.05:
                new_contradictions.append(Contradiction(
                    contradiction_type=ContradictionType.MISMATCH,
                    description=f'目的调整后仍有残余不匹配',
                    severity=adj['new_mismatch'] * 0.3,
                ))

        return {
            'success': success,
            'strategy': 'purpose_reevaluation',
            'adjustments': adjustments,
            'new_contradictions': [
                {'id': c.id, 'severity': c.severity} for c in new_contradictions
            ],
        }

    # ==================== Level 4: 妥协层 ====================

    def _resolve_compromise(
        self,
        contradiction: Contradiction,
        connections: List[Connection],
    ) -> Dict[str, Any]:
        """
        Level 4: 矛盾妥协。

        6.5 原理: 智能其实没有真正解决矛盾，只是让矛盾和其他矛盾妥协。
        在这一层，智能接受矛盾的存在，但通过调整使其影响最小化。

        策略:
        - 同时调整实际度和期望度，使它们在中间值相遇
        - 降低矛盾的整体严重度
        - 将矛盾标记为"已接受"
        """
        adjustments = []

        for conn in contradiction.involved_connections:
            if conn.expected_degree is not None:
                # 实际和期望在中间值妥协
                midpoint = (conn.degree + conn.expected_degree) / 2
                old_degree = conn.degree
                old_expected = conn.expected_degree

                conn.degree = old_degree + (midpoint - old_degree) * 0.5
                conn.expected_degree = old_expected + (midpoint - old_expected) * 0.5

                adjustments.append({
                    'connection_id': conn.id,
                    'old_degree': old_degree,
                    'new_degree': conn.degree,
                    'old_expected': old_expected,
                    'new_expected': conn.expected_degree,
                    'compromise_point': midpoint,
                })

        # 降低严重度
        contradiction.adjust_severity(contradiction.severity * 0.2)
        contradiction.mark_resolved('compromise', {'adjustments': adjustments})

        return {
            'success': True,
            'strategy': 'compromise',
            'adjustments': adjustments,
            'new_contradictions': [],
            'note': '矛盾通过妥协解决，双方都做了让步',
        }

    # ==================== 批量解决 ====================

    def resolve_all(
        self,
        contradictions: List[Contradiction],
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> List[Dict[str, Any]]:
        """
        分层解决所有矛盾。

        按严重度排序，优先解决最严重的矛盾。
        解决过程中产生的新矛盾也被处理。
        """
        # 按严重度排序
        sorted_contras = sorted(contradictions, key=lambda c: c.severity, reverse=True)

        results = []
        remaining = list(sorted_contras)

        max_rounds = 3  # 最多3轮（处理新产生的矛盾）

        for round_num in range(max_rounds):
            if not remaining:
                break

            current_batch = remaining
            remaining = []

            for contra in current_batch:
                if contra.resolved:
                    continue

                result = self.resolve(contra, things, connections)

                # 收集新产生的矛盾
                new_contra_info = result.get('new_contradictions', [])
                for nc_info in new_contra_info:
                    # 创建新的矛盾对象（简化版）
                    new_contra = Contradiction(
                        contradiction_type=ContradictionType.MISMATCH,
                        description=f'解决 {contra.id} 时产生的新矛盾',
                        severity=nc_info.get('severity', 0.1),
                    )
                    remaining.append(new_contra)

                results.append(result)

        return results

    @property
    def resolution_log(self) -> List[Dict[str, Any]]:
        return self._resolution_log.copy()

    def level_statistics(self) -> Dict[str, int]:
        """统计各层次解决矛盾的数量。"""
        stats: Dict[str, int] = {}
        for entry in self._resolution_log:
            if entry.get('success', False):
                level_name = entry.get('level_name', 'UNKNOWN')
                stats[level_name] = stats.get(level_name, 0) + 1
        return stats

    def __repr__(self) -> str:
        return f"<HierarchicalResolver: {len(self._resolution_log)} resolutions>"
