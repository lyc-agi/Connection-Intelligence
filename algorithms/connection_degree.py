from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from core.thing import Thing
from core.connection import Connection, ConnectionType


class ConnectionDegreeCalculator:
    """
    连接度计算器 - 基于事物属性和已有连接计算连接度。

    连接度的计算基于以下原则:
    1. 属性相似度 (直接比较两个事物的属性)
    2. 已有连接的传递性 (如果 A 连接 B，B 连接 C，则 A 和 C 有间接连接)
    3. 连接类型的权重 (不同类型的连接对度的贡献不同)
    4. 时间衰减 (历史连接度随时间减弱)
    """

    def __init__(self, decay_factor: float = 0.95, transitive_weight: float = 0.5):
        self.decay_factor = decay_factor
        self.transitive_weight = transitive_weight
        self._cache: Dict[str, float] = {}

    def compute_degree(
        self,
        thing_a: Thing,
        thing_b: Thing,
        existing_connections: List[Connection],
    ) -> float:
        """
        计算两个事物之间的连接度。

        综合考虑:
        - 直接连接
        - 属性相似度
        - 传递连接
        """
        direct_degree = self._direct_degree(thing_a, thing_b, existing_connections)
        attribute_degree = self._attribute_degree(thing_a, thing_b)
        transitive_degree = self._transitive_degree(thing_a, thing_b, existing_connections)

        w_direct = 0.5
        w_attr = 0.3
        w_trans = 0.2

        degree = (
            w_direct * direct_degree +
            w_attr * attribute_degree +
            w_trans * transitive_degree
        )

        return max(0.0, min(1.0, degree))

    def _direct_degree(
        self,
        thing_a: Thing,
        thing_b: Thing,
        connections: List[Connection],
    ) -> float:
        max_degree = 0.0
        for conn in connections:
            if conn.is_between(thing_a.id, thing_b.id):
                weighted = conn.degree * conn.weight
                max_degree = max(max_degree, weighted)
        return max_degree

    def _attribute_degree(self, thing_a: Thing, thing_b: Thing) -> float:
        return thing_a.similarity(thing_b)

    def _transitive_degree(
        self,
        thing_a: Thing,
        thing_b: Thing,
        connections: List[Connection],
    ) -> float:
        """
        计算通过中间事物的传递连接度。
        使用图的广度优先搜索寻找两跳路径。
        """
        adj: Dict[str, List[Tuple[str, float]]] = {}
        for conn in connections:
            adj.setdefault(conn.source_id, []).append((conn.target_id, conn.degree))
            adj.setdefault(conn.target_id, []).append((conn.source_id, conn.degree))

        neighbors_a = adj.get(thing_a.id, [])
        neighbors_b = adj.get(thing_b.id, [])

        if not neighbors_a or not neighbors_b:
            return 0.0

        b_set = set()
        for n_id, n_degree in neighbors_b:
            b_set.add(n_id)

        transitive_score = 0.0
        for n_id, n_degree in neighbors_a:
            if n_id in b_set:
                b_degree = 0.0
                for n2_id, n2_degree in neighbors_b:
                    if n2_id == n_id:
                        b_degree = max(b_degree, n2_degree)
                transitive_score = max(transitive_score, n_degree * b_degree)

        return transitive_score * self.transitive_weight

    def predict_connection(
        self,
        thing_a: Thing,
        thing_b: Thing,
        all_things: Dict[str, Thing],
        all_connections: List[Connection],
    ) -> float:
        """
        预测两个事物之间应该存在的连接度。
        这是"期望连接度"的计算基础。
        """
        actual = self.compute_degree(thing_a, thing_b, all_connections)

        # 基于全局连接模式预测期望度
        expected = self._expected_degree(thing_a, thing_b, all_things, all_connections)

        return 0.5 * actual + 0.5 * expected

    def _expected_degree(
        self,
        thing_a: Thing,
        thing_b: Thing,
        all_things: Dict[str, Thing],
        all_connections: List[Connection],
    ) -> float:
        """
        基于事物的属性模式和已有连接模式，计算期望连接度。
        """
        # 规则1: 属性相似的事物应该有较强连接
        similarity = thing_a.similarity(thing_b)

        # 规则2: 具有相似连接模式的事物应该有较强连接
        pattern_similarity = self._connection_pattern_similarity(
            thing_a, thing_b, all_connections
        )

        # 规则3: 处于相同连接类型的事物应该有较强连接
        type_compatibility = self._type_compatibility(thing_a, thing_b, all_connections)

        return 0.4 * similarity + 0.35 * pattern_similarity + 0.25 * type_compatibility

    def _connection_pattern_similarity(
        self,
        thing_a: Thing,
        thing_b: Thing,
        connections: List[Connection],
    ) -> float:
        a_connections = set()
        b_connections = set()

        for conn in connections:
            if conn.source_id == thing_a.id:
                a_connections.add(conn.target_id)
            if conn.target_id == thing_a.id:
                a_connections.add(conn.source_id)
            if conn.source_id == thing_b.id:
                b_connections.add(conn.target_id)
            if conn.target_id == thing_b.id:
                b_connections.add(conn.source_id)

        if not a_connections and not b_connections:
            return 0.0

        overlap = a_connections & b_connections
        union = a_connections | b_connections

        if not union:
            return 0.0

        return len(overlap) / len(union)

    def _type_compatibility(
        self,
        thing_a: Thing,
        thing_b: Thing,
        connections: List[Connection],
    ) -> float:
        a_types = set()
        b_types = set()

        for conn in connections:
            if conn.source_id == thing_a.id or conn.target_id == thing_a.id:
                a_types.add(conn.connection_type)
            if conn.source_id == thing_b.id or conn.target_id == thing_b.id:
                b_types.add(conn.connection_type)

        if not a_types and not b_types:
            return 0.0

        overlap = a_types & b_types
        union = a_types | b_types

        if not union:
            return 0.0

        return len(overlap) / len(union)

    def compute_network_strain(self, connections: List[Connection]) -> float:
        """
        计算整个连接网络的总应变。
        应变越大，说明网络中的矛盾越多。
        """
        return sum(c.strain() for c in connections)

    def compute_network_entropy(self, connections: List[Connection]) -> float:
        """
        计算连接度的信息熵。
        熵低说明连接度分布集中（要么高要么低），熵高说明分布均匀。
        """
        if not connections:
            return 0.0

        degrees = [c.degree for c in connections]
        n = len(degrees)
        if n == 0:
            return 0.0

        bins = [0] * 10
        for d in degrees:
            idx = min(int(d * 10), 9)
            bins[idx] += 1

        entropy = 0.0
        for count in bins:
            if count > 0:
                p = count / n
                entropy -= p * math.log2(p)

        return entropy

    def reset_cache(self) -> None:
        self._cache.clear()
