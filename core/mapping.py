from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from core.thing import Thing
from core.connection import Connection, ConnectionType
from core.contradiction import Contradiction, ContradictionType


class IsomorphismMapper:
    """
    同构映射器 - 实现 6.1 和 6.2 原理。

    6.1 外部矛盾→内部矛盾转换:
       只有当外部矛盾被转换为智能自身相关的矛盾时，智能才有动力去解决。

    6.2 映射/同构:
       智能在一定程度上映射了将要解决的对象的矛盾。
       载体和干预对象都是宇宙的产物，具有相似性，是映射实现的基础。

    核心功能:
    1. 检测两个子图之间的结构同构
    2. 将外部矛盾映射为智能内部的矛盾
    3. 建立外部事物到内部事物的对应关系
    """

    def __init__(self, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold
        self._mappings: List[Dict] = []  # 已建立的映射历史

    def find_isomorphism(
        self,
        external_things: Dict[str, Thing],
        external_connections: List[Connection],
        internal_things: Dict[str, Thing],
        internal_connections: List[Connection],
    ) -> Optional[Dict[str, str]]:
        """
        查找外部网络到内部网络的结构同构映射。

        返回 {external_thing_id: internal_thing_id} 映射，或 None。
        同构要求:
        1. 属性相似度超过阈值
        2. 连接结构一致 (如果 A-B 有连接，则 f(A)-f(B) 也有连接)
        3. 连接类型一致
        """
        # 步骤1: 基于属性相似度建立候选映射
        candidates = self._build_candidates(external_things, internal_things)

        if not candidates:
            return None

        # 步骤2: 验证结构一致性
        mapping = self._verify_structural_consistency(
            candidates, external_connections, internal_connections
        )

        if mapping:
            self._mappings.append({
                'type': 'isomorphism',
                'mapping': mapping,
                'external_size': len(external_things),
                'internal_size': len(internal_things),
            })

        return mapping if mapping else None

    def _build_candidates(
        self,
        external_things: Dict[str, Thing],
        internal_things: Dict[str, Thing],
    ) -> Dict[str, List[str]]:
        """
        为每个外部事物建立候选的内部事物列表。
        基于属性相似度。
        """
        candidates: Dict[str, List[str]] = {}

        for ext_id, ext_thing in external_things.items():
            candidates[ext_id] = []
            for int_id, int_thing in internal_things.items():
                sim = ext_thing.similarity(int_thing)
                if sim >= self.similarity_threshold:
                    candidates[ext_id].append((int_id, sim))

            # 按相似度排序
            candidates[ext_id].sort(key=lambda x: x[1], reverse=True)
            # 只保留 thing_id
            candidates[ext_id] = [tid for tid, _ in candidates[ext_id]]

        return candidates

    def _verify_structural_consistency(
        self,
        candidates: Dict[str, List[str]],
        external_connections: List[Connection],
        internal_connections: List[Connection],
    ) -> Optional[Dict[str, str]]:
        """
        验证候选映射的结构一致性。
        使用回溯法寻找一致的结构同构。
        """
        ext_ids = list(candidates.keys())

        # 构建内部连接的快速查找
        internal_conn_set: Set[Tuple[str, str, str]] = set()  # (src, tgt, type)
        for conn in internal_connections:
            internal_conn_set.add((conn.source_id, conn.target_id, conn.connection_type.value))
            internal_conn_set.add((conn.target_id, conn.source_id, conn.connection_type.value))

        external_conn_pairs: Set[Tuple[str, str, str]] = set()
        for conn in external_connections:
            external_conn_pairs.add((conn.source_id, conn.target_id, conn.connection_type.value))
            external_conn_pairs.add((conn.target_id, conn.source_id, conn.connection_type.value))

        # 回溯搜索
        mapping: Dict[str, str] = {}
        used_internal: Set[str] = set()

        def backtrack(idx: int) -> bool:
            if idx == len(ext_ids):
                return True

            ext_id = ext_ids[idx]
            for int_id in candidates.get(ext_id, []):
                if int_id in used_internal:
                    continue

                # 检查结构一致性
                valid = True
                for mapped_ext, mapped_int in mapping.items():
                    # 检查 ext_id - mapped_ext 的连接是否在内部有对应
                    for et, mt in [(ext_id, mapped_ext), (mapped_ext, ext_id)]:
                        for conn_type_str in {c[2] for c in external_conn_pairs if c[0] == et and c[1] == mt}:
                            it, mit = int_id, mapped_int
                            if (it, mit, conn_type_str) not in internal_conn_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break

                if valid:
                    mapping[ext_id] = int_id
                    used_internal.add(int_id)

                    if backtrack(idx + 1):
                        return True

                    del mapping[ext_id]
                    used_internal.discard(int_id)

            return False

        if backtrack(0):
            return mapping
        return None

    def convert_external_contradiction(
        self,
        external_contradiction: Contradiction,
        external_things: Dict[str, Thing],
        internal_things: Dict[str, Thing],
        internal_connections: List[Connection],
        intelligence_self_id: Optional[str] = None,
    ) -> Optional[Contradiction]:
        """
        将外部矛盾转换为智能自身的内部矛盾。

        6.1 原理: 外部矛盾被转换为智能自身相关的矛盾。
        例如: 机器故障 -> 工人收入减少

        步骤:
        1. 找到外部事物到内部事物的映射
        2. 将矛盾涉及的连接映射到内部
        3. 确保内部矛盾与智能自身相关 (通过 intelligence_self_id)
        """
        # 提取外部矛盾涉及的连接对应的事物
        external_involved_things: Dict[str, Thing] = {}
        for conn in external_contradiction.involved_connections:
            if conn.source_id in external_things:
                external_involved_things[conn.source_id] = external_things[conn.source_id]
            if conn.target_id in external_things:
                external_involved_things[conn.target_id] = external_things[conn.target_id]

        if not external_involved_things:
            return None

        # 查找同构映射
        mapping = self.find_isomorphism(
            external_involved_things,
            external_contradiction.involved_connections,
            internal_things,
            internal_connections,
        )

        if not mapping:
            # 如果找不到完全同构，使用属性相似度找最佳匹配
            mapping = self._find_best_matches(external_involved_things, internal_things)

        if not mapping:
            return None

        # 确保映射包含智能自身
        if intelligence_self_id:
            # 如果智能自身不在映射中，将最关键的外部事物映射到智能自身
            if intelligence_self_id not in mapping.values():
                # 找到外部矛盾中最严重的事物
                most_critical = self._find_most_critical_external(
                    external_contradiction, external_things
                )
                if most_critical:
                    mapping[most_critical] = intelligence_self_id

        # 构建内部矛盾
        internal_conns = []
        for ext_conn in external_contradiction.involved_connections:
            int_src = mapping.get(ext_conn.source_id)
            int_tgt = mapping.get(ext_conn.target_id)

            if int_src and int_tgt:
                # 查找或创建对应的内部连接
                found_conn = None
                for conn in internal_connections:
                    if conn.is_between(int_src, int_tgt):
                        found_conn = conn
                        break

                if found_conn:
                    # 如果已有内部连接，将外部矛盾的期望度传递给它
                    if ext_conn.expected_degree is not None and found_conn.expected_degree is None:
                        found_conn.expected_degree = ext_conn.expected_degree
                    internal_conns.append(found_conn)
                else:
                    # 创建一个新的内部连接表示这个矛盾
                    new_conn = Connection(
                        source_id=int_src,
                        target_id=int_tgt,
                        degree=0.3,
                        connection_type=ext_conn.connection_type,
                        expected_degree=ext_conn.expected_degree,
                    )
                    internal_conns.append(new_conn)

        if not internal_conns:
            return None

        # 创建内部矛盾
        internal_contradiction = Contradiction(
            contradiction_type=external_contradiction.contradiction_type,
            description=f"[内部化] {external_contradiction.description}",
            involved_connections=internal_conns,
            severity=external_contradiction.severity * 0.8,  # 内部化后严重度略降
            context={
                'source': 'external_conversion',
                'external_contradiction_id': external_contradiction.id,
                'mapping': mapping,
                'intelligence_self_id': intelligence_self_id,
            },
        )

        self._mappings.append({
            'type': 'contradiction_conversion',
            'external_id': external_contradiction.id,
            'internal_id': internal_contradiction.id,
            'mapping': mapping,
        })

        return internal_contradiction

    def _find_best_matches(
        self,
        external_things: Dict[str, Thing],
        internal_things: Dict[str, Thing],
    ) -> Dict[str, str]:
        """找不到完全同构时，使用贪心策略找最佳匹配。"""
        mapping: Dict[str, str] = {}
        used: Set[str] = set()

        # 按属性数量排序，属性多的优先匹配
        ext_sorted = sorted(
            external_things.items(),
            key=lambda x: len(x[1].attributes),
            reverse=True
        )

        for ext_id, ext_thing in ext_sorted:
            best_match = None
            best_sim = 0.0

            for int_id, int_thing in internal_things.items():
                if int_id in used:
                    continue
                sim = ext_thing.similarity(int_thing)
                if sim > best_sim:
                    best_sim = sim
                    best_match = int_id

            if best_match and best_sim > 0:
                mapping[ext_id] = best_match
                used.add(best_match)

        return mapping

    def _find_most_critical_external(
        self,
        contradiction: Contradiction,
        things: Dict[str, Thing],
    ) -> Optional[str]:
        """找到外部矛盾中最关键的事物（连接最多或最中心）。"""
        thing_conn_count: Dict[str, int] = {}
        for conn in contradiction.involved_connections:
            thing_conn_count[conn.source_id] = thing_conn_count.get(conn.source_id, 0) + 1
            thing_conn_count[conn.target_id] = thing_conn_count.get(conn.target_id, 0) + 1

        if not thing_conn_count:
            return None

        return max(thing_conn_count, key=thing_conn_count.get)

    def structural_similarity(
        self,
        things_a: Dict[str, Thing],
        connections_a: List[Connection],
        things_b: Dict[str, Thing],
        connections_b: List[Connection],
    ) -> float:
        """
        计算两个网络的结构相似度 (0-1)。
        不要求完全同构，而是计算结构特征的相似度。
        """
        # 特征1: 节点数比
        size_ratio = min(len(things_a), len(things_b)) / max(len(things_a), len(things_b), 1)

        # 特征2: 边数比
        edge_ratio = min(len(connections_a), len(connections_b)) / max(len(connections_a), len(connections_b), 1)

        # 特征3: 连接类型分布相似度
        type_sim = self._type_distribution_similarity(connections_a, connections_b)

        # 特征4: 平均度数比
        avg_a = sum(c.degree for c in connections_a) / max(len(connections_a), 1)
        avg_b = sum(c.degree for c in connections_b) / max(len(connections_b), 1)
        degree_ratio = min(avg_a, avg_b) / max(avg_a, avg_b, 0.001)

        # 特征5: 同构匹配率
        mapping = self.find_isomorphism(things_a, connections_a, things_b, connections_b)
        iso_rate = len(mapping) / max(len(things_a), 1) if mapping else 0.0

        # 加权综合
        return (
            0.15 * size_ratio +
            0.15 * edge_ratio +
            0.2 * type_sim +
            0.2 * degree_ratio +
            0.3 * iso_rate
        )

    def _type_distribution_similarity(
        self,
        conns_a: List[Connection],
        conns_b: List[Connection],
    ) -> float:
        """计算连接类型分布的相似度。"""
        if not conns_a and not conns_b:
            return 1.0

        types_a: Dict[str, float] = {}
        types_b: Dict[str, float] = {}

        for c in conns_a:
            t = c.connection_type.value
            types_a[t] = types_a.get(t, 0) + 1
        for c in conns_b:
            t = c.connection_type.value
            types_b[t] = types_b.get(t, 0) + 1

        total_a = sum(types_a.values())
        total_b = sum(types_b.values())

        all_types = set(types_a.keys()) | set(types_b.keys())
        similarity = 0.0
        for t in all_types:
            pa = types_a.get(t, 0) / max(total_a, 1)
            pb = types_b.get(t, 0) / max(total_b, 1)
            similarity += min(pa, pb)

        return similarity

    @property
    def mapping_history(self) -> List[Dict]:
        return self._mappings.copy()

    def __repr__(self) -> str:
        return f"<IsomorphismMapper: {len(self._mappings)} mappings>"
