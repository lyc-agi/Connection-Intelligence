from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from .thing import Thing
from .connection import Connection, ConnectionType
from .contradiction import Contradiction, ContradictionType


class Intelligence:
    """
    智能引擎 - 协调万物之间的连接度。

    智能的本质是解决矛盾，而矛盾的本质在于连接程度的不匹配。
    智能通过以下方式发挥作用:
    1. 感知连接网络中的矛盾
    2. 将外部矛盾转换为自身相关的矛盾
    3. 通过调整连接度或创造新连接来解决矛盾
    4. 利用规律（连接类型）实现目的
    5. 从更高层次审视矛盾，选择最优解决方案
    """

    def __init__(self, name: str = "Intelligence", learning_rate: float = 0.1):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.learning_rate = learning_rate

        self._things: Dict[str, Thing] = {}
        self._connections: Dict[str, Connection] = {}
        self._contradictions: List[Contradiction] = []
        self._history: List[Dict[str, Any]] = []

        self._resolution_strategies: Dict[str, Callable] = {
            'adjust_degree': self._strategy_adjust_degree,
            'create_connection': self._strategy_create_connection,
            'remove_connection': self._strategy_remove_connection,
            'reinterpret': self._strategy_reinterpret,
        }

        self._external_constraints: List[Dict[str, Any]] = []
        self._objective_function: Optional[Callable] = None

    # ==================== 事物管理 ====================

    def add_thing(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Thing:
        thing = Thing(name, attributes)
        self._things[thing.id] = thing
        return thing

    def get_thing(self, thing_id: str) -> Optional[Thing]:
        return self._things.get(thing_id)

    def remove_thing(self, thing_id: str) -> bool:
        thing = self._things.pop(thing_id, None)
        if thing:
            conns_to_remove = []
            for conn_id, conn in self._connections.items():
                if conn.source_id == thing_id or conn.target_id == thing_id:
                    conns_to_remove.append(conn_id)
            for conn_id in conns_to_remove:
                self._connections.pop(conn_id, None)
            return True
        return False

    @property
    def things(self) -> List[Thing]:
        return list(self._things.values())

    # ==================== 连接管理 ====================

    def add_connection(
        self,
        source_id: str,
        target_id: str,
        degree: float = 0.5,
        connection_type: ConnectionType = ConnectionType.CUSTOM,
        weight: float = 1.0,
        expected_degree: Optional[float] = None,
    ) -> Optional[Connection]:
        if source_id not in self._things or target_id not in self._things:
            return None

        conn = Connection(source_id, target_id, degree, connection_type, weight)
        if expected_degree is not None:
            conn.expected_degree = expected_degree

        self._connections[conn.id] = conn
        self._things[source_id].register_connection(conn.id, 'out')
        self._things[target_id].register_connection(conn.id, 'in')
        return conn

    def get_connection(self, connection_id: str) -> Optional[Connection]:
        return self._connections.get(connection_id)

    def find_between(self, thing_a_id: str, thing_b_id: str) -> List[Connection]:
        result = []
        for conn in self._connections.values():
            if conn.is_between(thing_a_id, thing_b_id):
                result.append(conn)
        return result

    def remove_connection(self, connection_id: str) -> bool:
        conn = self._connections.pop(connection_id, None)
        if conn:
            thing_a = self._things.get(conn.source_id)
            thing_b = self._things.get(conn.target_id)
            if thing_a:
                thing_a.unregister_connection(connection_id)
            if thing_b:
                thing_b.unregister_connection(connection_id)
            return True
        return False

    @property
    def connections(self) -> List[Connection]:
        return list(self._connections.values())

    # ==================== 矛盾检测 ====================

    def detect_contradictions(self) -> List[Contradiction]:
        """
        检测当前连接网络中所有的矛盾。

        矛盾来源:
        1. 连接度不匹配 (期望 vs 实际)
        2. 约束违反
        3. 循环依赖
        4. 冲突需求
        """
        new_contradictions: List[Contradiction] = []

        for conn in self._connections.values():
            if conn.mismatch > 0.01:
                contradiction = Contradiction(
                    contradiction_type=ContradictionType.MISMATCH,
                    description=f"连接 {conn.id} 度不匹配: 实际={conn.degree:.3f}, "
                                f"期望={conn.expected_degree:.3f}, 差异={conn.mismatch:.3f}",
                    involved_connections=[conn],
                    severity=conn.mismatch,
                )
                new_contradictions.append(contradiction)

            if not conn.satisfies_constraints():
                total_violation = 0.0
                for c in conn.constraints:
                    if c['type'] == 'range':
                        lo = c.get('min', 0.0)
                        hi = c.get('max', 1.0)
                        if conn.degree < lo:
                            total_violation += (lo - conn.degree)
                        elif conn.degree > hi:
                            total_violation += (conn.degree - hi)

                contradiction = Contradiction(
                    contradiction_type=ContradictionType.CONSTRAINT_VIOLATION,
                    description=f"连接 {conn.id} 违反约束",
                    involved_connections=[conn],
                    severity=min(total_violation, 1.0),
                )
                new_contradictions.append(contradiction)

        cycle_contradictions = self._detect_circular_dependencies()
        new_contradictions.extend(cycle_contradictions)

        conflicting = self._detect_conflicting_demands()
        new_contradictions.extend(conflicting)

        self._contradictions = [c for c in self._contradictions if c.detect()]
        self._contradictions.extend(new_contradictions)
        return self._contradictions

    def _detect_circular_dependencies(self) -> List[Contradiction]:
        contradictions = []
        graph: Dict[str, List[str]] = {}

        for conn in self._connections.values():
            if conn.connection_type == ConnectionType.DEPENDENCY:
                graph.setdefault(conn.source_id, []).append(conn.target_id)

        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    contradictions.append(Contradiction(
                        contradiction_type=ContradictionType.CIRCULAR_DEPENDENCY,
                        description=f"检测到循环依赖，始于节点 {node}",
                        severity=0.7,
                        context={'has_cycle': True, 'start_node': node},
                    ))
                    break

        return contradictions

    def _detect_conflicting_demands(self) -> List[Contradiction]:
        contradictions = []
        things_with_multiple = self._find_things_with_conflicting_connections()

        for thing_id, conflicting_conns in things_with_multiple.items():
            if len(conflicting_conns) >= 2:
                contradictions.append(Contradiction(
                    contradiction_type=ContradictionType.CONFLICTING_DEMANDS,
                    description=f"事物 {thing_id} 存在 {len(conflicting_conns)} 个冲突连接需求",
                    involved_connections=conflicting_conns[:3],
                    severity=0.5 + 0.1 * min(len(conflicting_conns), 5),
                ))

        return contradictions

    def _find_things_with_conflicting_connections(self) -> Dict[str, List[Connection]]:
        conflicts: Dict[str, List[Connection]] = {}
        for thing_id in self._things:
            out_conns = [c for c in self._connections.values() if c.source_id == thing_id]
            mismatched = [c for c in out_conns if c.mismatch > 0.1]
            if len(mismatched) >= 2:
                conflicts[thing_id] = mismatched
        return conflicts

    @property
    def active_contradictions(self) -> List[Contradiction]:
        return [c for c in self._contradictions if not c.resolved]

    # ==================== 矛盾解决 ====================

    def resolve_contradictions(
        self,
        max_iterations: int = 50,
        strategy: str = 'auto',
    ) -> List[Dict[str, Any]]:
        """
        解决当前所有活跃的矛盾。

        返回每个矛盾的解决记录。
        """
        results = []

        for iteration in range(max_iterations):
            active = self.active_contradictions
            if not active:
                break

            contradiction = active[0]
            resolution = self._resolve_single(contradiction, strategy)
            results.append(resolution)

            self._history.append({
                'iteration': iteration,
                'action': 'resolve',
                'contradiction_id': contradiction.id,
                'resolution': resolution,
            })

        return results

    def _resolve_single(
        self,
        contradiction: Contradiction,
        strategy: str,
    ) -> Dict[str, Any]:
        """
        解决单个矛盾。
        """
        if strategy == 'auto':
            strategy = self._select_strategy(contradiction)

        resolver = self._resolution_strategies.get(strategy)
        if resolver:
            return resolver(contradiction)

        return self._strategy_adjust_degree(contradiction)

    def _select_strategy(self, contradiction: Contradiction) -> str:
        """
        根据矛盾类型选择解决策略。
        """
        ctype = contradiction.contradiction_type

        if ctype == ContradictionType.MISMATCH:
            if contradiction.severity < 0.3:
                return 'adjust_degree'
            else:
                return 'create_connection'

        elif ctype == ContradictionType.CONSTRAINT_VIOLATION:
            return 'adjust_degree'

        elif ctype == ContradictionType.CIRCULAR_DEPENDENCY:
            return 'remove_connection'

        elif ctype == ContradictionType.CONFLICTING_DEMANDS:
            return 'reinterpret'

        elif ctype == ContradictionType.EXTERNAL_PRESSURE:
            return 'create_connection'

        return 'adjust_degree'

    def _strategy_adjust_degree(self, contradiction: Contradiction) -> Dict[str, Any]:
        """
        策略1: 调整连接度 - 通过逐步逼近目标值来解决不匹配。
        """
        adjustments = []
        for conn in contradiction.involved_connections:
            old_degree = conn.degree
            delta = conn.relax(self.learning_rate)
            adjustments.append({
                'connection_id': conn.id,
                'old_degree': old_degree,
                'new_degree': conn.degree,
                'delta': delta,
            })

        contradiction.mark_resolved('adjust_degree', {'adjustments': adjustments})
        return {
            'strategy': 'adjust_degree',
            'success': True,
            'adjustments': adjustments,
        }

    def _strategy_create_connection(self, contradiction: Contradiction) -> Dict[str, Any]:
        """
        策略2: 创造新连接 - 当现有连接无法解决矛盾时，创造新的连接。
        这对应了智能"跳出既定路线"的能力。
        """
        affected_things = set()
        for conn in contradiction.involved_connections:
            affected_things.add(conn.source_id)
            affected_things.add(conn.target_id)

        if len(affected_things) >= 2:
            thing_list = list(affected_things)
            new_conn = self.add_connection(
                source_id=thing_list[0],
                target_id=thing_list[1],
                degree=0.8,
                connection_type=ConnectionType.FUNCTIONAL,
                weight=1.5,
            )

            if new_conn:
                new_conn.expected_degree = 0.8
                contradiction.mark_resolved('create_connection', {
                    'new_connection_id': new_conn.id,
                    'between': (thing_list[0], thing_list[1]),
                })
                return {
                    'strategy': 'create_connection',
                    'success': True,
                    'new_connection': new_conn.id,
                }

        contradiction.adjust_severity(contradiction.severity * 0.5)
        return {'strategy': 'create_connection', 'success': False}

    def _strategy_remove_connection(self, contradiction: Contradiction) -> Dict[str, Any]:
        """
        策略3: 移除连接 - 对于循环依赖等情况，移除最弱的连接。
        """
        if not contradiction.involved_connections:
            contradiction.mark_resolved('remove_connection', {})
            return {'strategy': 'remove_connection', 'success': True}

        weakest = min(contradiction.involved_connections, key=lambda c: c.weight)
        success = self.remove_connection(weakest.id)

        contradiction.mark_resolved('remove_connection', {
            'removed_connection': weakest.id if success else None,
        })
        return {
            'strategy': 'remove_connection',
            'success': success,
            'removed': weakest.id if success else None,
        }

    def _strategy_reinterpret(self, contradiction: Contradiction) -> Dict[str, Any]:
        """
        策略4: 重新诠释 - 从更高层次审视矛盾，调整期望而非实际。
        这对应了智能"跳出矛盾所在层次"的能力。
        """
        adjustments = []
        for conn in contradiction.involved_connections:
            if conn.expected_degree is not None:
                old_expected = conn.expected_degree
                new_expected = conn.degree + (conn.expected_degree - conn.degree) * 0.3
                conn.expected_degree = new_expected
                adjustments.append({
                    'connection_id': conn.id,
                    'old_expected': old_expected,
                    'new_expected': new_expected,
                })

        contradiction.mark_resolved('reinterpret', {'adjustments': adjustments})
        return {
            'strategy': 'reinterpret',
            'success': True,
            'adjustments': adjustments,
        }

    # ==================== 智能度量 ====================

    def measure_intelligence(self, scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        度量智能程度。

        D = Q - P
        P: 无智能干预时矛盾解决的概率
        Q: 有智能干预时矛盾解决的概率

        返回 D 值及相关指标。
        """
        contradictions = self.active_contradictions
        if not contradictions:
            return {
                'D': 0.0,
                'P': 0.0,
                'Q': 0.0,
                'status': 'no_contradictions',
            }

        total_severity = sum(c.severity for c in contradictions)
        avg_severity = total_severity / len(contradictions) if contradictions else 0

        # P: 无干预时，矛盾自然缓解的概率
        # 基于自然界"弛豫"原理，矛盾有自发缓解的趋势
        P = avg_severity * 0.1

        # Q: 有干预时，矛盾被解决的概率
        # 基于历史解决率和当前策略有效性
        resolution_rate = self._historical_resolution_rate()
        Q = avg_severity * resolution_rate

        D = Q - P

        return {
            'D': D,
            'P': P,
            'Q': Q,
            'active_contradictions': len(contradictions),
            'avg_severity': avg_severity,
            'resolution_rate': resolution_rate,
            'total_things': len(self._things),
            'total_connections': len(self._connections),
        }

    def _historical_resolution_rate(self) -> float:
        if not self._history:
            return 0.5
        resolutions = [h for h in self._history if h.get('action') == 'resolve']
        if not resolutions:
            return 0.5
        return min(len(resolutions) / max(len(self._history), 1), 1.0)

    # ==================== 外部约束 ====================

    def add_external_constraint(self, constraint: Dict[str, Any]) -> None:
        """
        添加外部约束，代表智能需要满足的外部规律或目的。
        """
        self._external_constraints.append(constraint)

    def set_objective(self, objective_fn: Callable) -> None:
        """
        设置智能的目标函数。
        """
        self._objective_function = objective_fn

    # ==================== 统计与状态 ====================

    @property
    def stats(self) -> Dict[str, Any]:
        resolved = [c for c in self._contradictions if c.resolved]
        active = [c for c in self._contradictions if not c.resolved]

        return {
            'name': self.name,
            'id': self.id,
            'things_count': len(self._things),
            'connections_count': len(self._connections),
            'total_contradictions': len(self._contradictions),
            'resolved_contradictions': len(resolved),
            'active_contradictions': len(active),
            'resolution_history_length': len(self._history),
            'learning_rate': self.learning_rate,
        }

    def network_summary(self) -> str:
        """
        生成连接网络的摘要描述。
        """
        lines = [f"=== 智能网络: {self.name} ==="]
        lines.append(f"事物数: {len(self._things)}")
        lines.append(f"连接数: {len(self._connections)}")
        lines.append(f"矛盾数: {len(self._contradictions)} (活跃: {len(self.active_contradictions)})")
        lines.append("")

        for thing in self._things.values():
            conns_in = [c for c in self._connections.values() if c.target_id == thing.id]
            conns_out = [c for c in self._connections.values() if c.source_id == thing.id]
            lines.append(f"  [{thing.name}] (入:{len(conns_in)} 出:{len(conns_out)})")

        lines.append("")
        for conn in self._connections.values():
            src = self._things.get(conn.source_id)
            tgt = self._things.get(conn.target_id)
            src_name = src.name if src else "?"
            tgt_name = tgt.name if tgt else "?"
            lines.append(f"  {src_name} --[{conn.connection_type.value}:{conn.degree:.3f}]--> {tgt_name}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Intelligence: {self.name} ({self.id}) things={len(self._things)} conns={len(self._connections)}>"
