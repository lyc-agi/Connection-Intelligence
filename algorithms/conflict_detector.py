from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from core.thing import Thing
from core.connection import Connection, ConnectionType
from core.contradiction import Contradiction, ContradictionType


class ConflictDetector:
    """
    冲突检测器 - 识别连接网络中的各种矛盾。

    检测类型:
    1. 连接度不匹配 (期望 vs 实际)
    2. 约束违反 (连接度超出约束范围)
    3. 循环依赖 (依赖关系形成环)
    4. 资源冲突 (多个连接竞争同一资源/事物)
    5. 层次冲突 (不同层次的连接要求不一致)
    """

    def __init__(self, mismatch_threshold: float = 0.05, conflict_threshold: float = 0.3):
        self.mismatch_threshold = mismatch_threshold
        self.conflict_threshold = conflict_threshold

    def detect_all(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> List[Contradiction]:
        """
        运行所有类型的冲突检测。
        """
        contradictions: List[Contradiction] = []

        contradictions.extend(self.detect_mismatches(connections))
        contradictions.extend(self.detect_constraint_violations(connections))
        contradictions.extend(self.detect_circular_dependencies(connections))
        contradictions.extend(self.detect_resource_conflicts(things, connections))
        contradictions.extend(self.detect_hierarchical_conflicts(connections))

        return contradictions

    def detect_mismatches(self, connections: List[Connection]) -> List[Contradiction]:
        """
        检测连接度不匹配。
        """
        result = []
        for conn in connections:
            mismatch = conn.mismatch
            if mismatch > self.mismatch_threshold:
                result.append(Contradiction(
                    contradiction_type=ContradictionType.MISMATCH,
                    description=(
                        f"连接 {conn.id} ({conn.source_id}->{conn.target_id}) "
                        f"度不匹配: 实际={conn.degree:.3f}, "
                        f"期望={conn.expected_degree:.3f}, 差异={mismatch:.3f}"
                    ),
                    involved_connections=[conn],
                    severity=mismatch,
                ))
        return result

    def detect_constraint_violations(self, connections: List[Connection]) -> List[Contradiction]:
        """
        检测约束违反。
        """
        result = []
        for conn in connections:
            if not conn.satisfies_constraints():
                violation_details = []
                for c in conn.constraints:
                    if c['type'] == 'range':
                        lo = c.get('min', 0.0)
                        hi = c.get('max', 1.0)
                        if conn.degree < lo:
                            violation_details.append(f"低于最小值 {lo:.2f} (当前: {conn.degree:.3f})")
                        elif conn.degree > hi:
                            violation_details.append(f"超过最大值 {hi:.2f} (当前: {conn.degree:.3f})")

                result.append(Contradiction(
                    contradiction_type=ContradictionType.CONSTRAINT_VIOLATION,
                    description=f"连接 {conn.id} 约束违反: {'; '.join(violation_details)}",
                    involved_connections=[conn],
                    severity=conn.strain(),
                ))
        return result

    def detect_circular_dependencies(self, connections: List[Connection]) -> List[Contradiction]:
        """
        检测循环依赖。
        """
        result = []

        graph: Dict[str, List[str]] = {}
        for conn in connections:
            graph.setdefault(conn.source_id, []).append(conn.target_id)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {}
        for node in graph:
            color[node] = WHITE

        cycles_found = []

        def dfs(node: str, path: List[str]) -> None:
            color[node] = GRAY
            for neighbor in graph.get(node, []):
                if neighbor not in color:
                    color[neighbor] = WHITE
                if color[neighbor] == GRAY:
                    cycle_start = path.index(neighbor) if neighbor in path else 0
                    cycles_found.append(path[cycle_start:] + [neighbor])
                elif color[neighbor] == WHITE:
                    dfs(neighbor, path + [neighbor])
            color[node] = BLACK

        for node in list(graph.keys()):
            if color.get(node, WHITE) == WHITE:
                dfs(node, [node])

        for cycle in cycles_found:
            cycle_conns = []
            for i in range(len(cycle) - 1):
                for conn in connections:
                    if conn.source_id == cycle[i] and conn.target_id == cycle[i + 1]:
                        cycle_conns.append(conn)

            result.append(Contradiction(
                contradiction_type=ContradictionType.CIRCULAR_DEPENDENCY,
                description=f"检测到循环依赖: {' -> '.join(cycle)}",
                involved_connections=cycle_conns,
                severity=0.7,
                context={'cycle': cycle},
            ))

        return result

    def detect_resource_conflicts(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> List[Contradiction]:
        """
        检测资源冲突 - 当多个高权重连接竞争同一事物时产生。
        """
        result = []

        for thing_id, thing in things.items():
            incoming = [c for c in connections if c.target_id == thing_id]
            outgoing = [c for c in connections if c.source_id == thing_id]

            if len(incoming) >= 2:
                high_degree_conns = [c for c in incoming if c.degree > self.conflict_threshold]
                if len(high_degree_conns) >= 2:
                    result.append(Contradiction(
                        contradiction_type=ContradictionType.CONFLICTING_DEMANDS,
                        description=(
                            f"事物 {thing.name} 存在资源冲突: "
                            f"{len(high_degree_conns)} 个高连接度输入"
                        ),
                        involved_connections=high_degree_conns[:3],
                        severity=min(len(high_degree_conns) * 0.2, 1.0),
                        context={'thing_id': thing_id, 'conflict_type': 'incoming'},
                    ))

            if len(outgoing) >= 2:
                high_degree_conns = [c for c in outgoing if c.degree > self.conflict_threshold]
                if len(high_degree_conns) >= 2:
                    result.append(Contradiction(
                        contradiction_type=ContradictionType.CONFLICTING_DEMANDS,
                        description=(
                            f"事物 {thing.name} 存在资源冲突: "
                            f"{len(high_degree_conns)} 个高连接度输出"
                        ),
                        involved_connections=high_degree_conns[:3],
                        severity=min(len(high_degree_conns) * 0.2, 1.0),
                        context={'thing_id': thing_id, 'conflict_type': 'outgoing'},
                    ))

        return result

    def detect_hierarchical_conflicts(self, connections: List[Connection]) -> List[Contradiction]:
        """
        检测层次冲突 - 当层次连接与功能连接不一致时。
        """
        result = []

        hierarchical = [c for c in connections if c.connection_type == ConnectionType.HIERARCHICAL]
        functional = [c for c in connections if c.connection_type == ConnectionType.FUNCTIONAL]

        for h_conn in hierarchical:
            for f_conn in functional:
                if h_conn.is_between(f_conn.source_id, f_conn.target_id):
                    if abs(h_conn.degree - f_conn.degree) > self.conflict_threshold:
                        result.append(Contradiction(
                            contradiction_type=ContradictionType.CONFLICTING_DEMANDS,
                            description=(
                                f"层次/功能连接冲突: {h_conn.id} (层次={h_conn.degree:.3f}) vs "
                                f"{f_conn.id} (功能={f_conn.degree:.3f})"
                            ),
                            involved_connections=[h_conn, f_conn],
                            severity=abs(h_conn.degree - f_conn.degree),
                        ))

        return result
