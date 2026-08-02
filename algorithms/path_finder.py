from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from core.thing import Thing
from core.connection import Connection, ConnectionType
from core.network import NetworkGraph


class PathFinder:
    """
    路径搜索器 - 实现 6.3 原理。

    6.3 原理: 智能创造新的连接。
    智能之所以能够解决矛盾，是因为具有"跳出既定路线"的能力，
    也就是找到了一条新的路线，创造了原本不存在的连接。

    核心功能:
    1. 在连接网络中搜索新路径
    2. 评估路径的质量（连接度、代价、可靠性）
    3. 建议最优的新连接来桥接矛盾
    4. 发现潜在的间接连接路径
    """

    def __init__(self, network: Optional[NetworkGraph] = None):
        self.network = network or NetworkGraph()
        self._discovered_paths: List[Dict] = []

    def find_new_connection_route(
        self,
        source_id: str,
        target_id: str,
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> Optional[Dict]:
        """
        为两个缺少直接连接（或连接度不足）的事物寻找新的连接路径。

        这是"智能创造新连接"的核心算法:
        1. 检查是否已有直接连接
        2. 搜索间接路径
        3. 评估最佳路径
        4. 建议创建新连接

        返回路径建议，或 None。
        """
        conn_dict = {c.id: c for c in connections}
        self.network.build_from(things, conn_dict)

        # 检查现有直接连接
        existing = self._find_direct_connection(source_id, target_id, connections)

        # 搜索间接路径
        all_paths = self.network.find_all_paths(source_id, target_id, max_depth=4)

        # 搜索桥梁节点
        bridges = self.network.find_bridge_nodes(source_id, target_id)

        # 搜索缺失连接
        missing = self.network.find_missing_connections(threshold=0.2)
        relevant_missing = [
            m for m in missing
            if (m[0] == source_id and m[1] == target_id) or
               (m[0] == target_id and m[1] == source_id)
        ]

        # 评估所有候选路径
        candidates = self._evaluate_candidates(
            source_id, target_id, all_paths, bridges, relevant_missing,
            things, connections
        )

        if not candidates:
            return None

        # 选择最佳路径
        best = candidates[0]

        result = {
            'source_id': source_id,
            'target_id': target_id,
            'existing_connection': {
                'id': existing.id,
                'degree': existing.degree,
                'expected': existing.expected_degree,
            } if existing else None,
            'best_route': best,
            'all_candidates': candidates[:3],  # 返回前3个候选
            'recommendation': self._make_recommendation(best, existing),
        }

        self._discovered_paths.append(result)
        return result

    def _find_direct_connection(
        self,
        a_id: str,
        b_id: str,
        connections: List[Connection],
    ) -> Optional[Connection]:
        """查找直接连接。"""
        for conn in connections:
            if conn.is_between(a_id, b_id):
                return conn
        return None

    def _evaluate_candidates(
        self,
        source_id: str,
        target_id: str,
        all_paths: List[List[str]],
        bridges: List[str],
        missing: List[Tuple[str, str, float]],
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> List[Dict]:
        """
        评估所有候选路径，返回按质量排序的列表。
        """
        candidates: List[Dict] = []

        # 候选1: 直接创建新连接
        thing_a = things.get(source_id)
        thing_b = things.get(target_id)
        if thing_a and thing_b:
            attr_sim = thing_a.similarity(thing_b)
            suggested_degree = max(0.3, attr_sim)
            candidates.append({
                'type': 'direct_new',
                'path': [source_id, target_id],
                'suggested_degree': suggested_degree,
                'cost': 1.0 - suggested_degree,
                'reliability': attr_sim,
                'description': f'直接创建 {thing_a.name} -> {thing_b.name} 的新连接',
            })

        # 候选2: 通过桥梁节点的间接路径
        for bridge_id in bridges:
            bridge_thing = things.get(bridge_id)
            bridge_name = bridge_thing.name if bridge_thing else bridge_id

            # 计算路径质量
            conn_a_bridge = self._find_direct_connection(source_id, bridge_id, connections)
            conn_bridge_b = self._find_direct_connection(bridge_id, target_id, connections)

            if conn_a_bridge and conn_bridge_b:
                path_degree = min(conn_a_bridge.degree, conn_bridge_b.degree)
                cost = (1 - conn_a_bridge.degree) + (1 - conn_bridge_b.degree)
                reliability = path_degree * 0.8  # 间接路径可靠性略低
            else:
                path_degree = 0.3
                cost = 1.4
                reliability = 0.2

            candidates.append({
                'type': 'via_bridge',
                'path': [source_id, bridge_id, target_id],
                'bridge_node': bridge_id,
                'bridge_name': bridge_name,
                'suggested_degree': path_degree,
                'cost': cost,
                'reliability': reliability,
                'description': f'通过桥梁节点 {bridge_name} 建立间接连接',
            })

        # 候选3: 多跳路径
        for path in all_paths:
            if len(path) < 3 or len(path) > 5:
                continue
            if path in [c['path'] for c in candidates]:
                continue

            # 计算路径上所有连接的最小度
            min_degree = 1.0
            total_cost = 0.0
            valid = True

            for i in range(len(path) - 1):
                conn = self._find_direct_connection(path[i], path[i + 1], connections)
                if conn:
                    min_degree = min(min_degree, conn.degree)
                    total_cost += 1 - conn.degree
                else:
                    valid = False
                    break

            if valid and min_degree > 0:
                path_names = [things.get(tid).name if things.get(tid) else tid for tid in path]
                candidates.append({
                    'type': 'multi_hop',
                    'path': path,
                    'path_names': path_names,
                    'suggested_degree': min_degree,
                    'cost': total_cost,
                    'reliability': min_degree * (0.7 ** (len(path) - 2)),
                    'description': f'多跳路径: {" -> ".join(path_names)}',
                })

        # 按综合质量排序: reliability / cost
        candidates.sort(key=lambda c: c['reliability'] / max(c['cost'], 0.01), reverse=True)

        return candidates

    def _make_recommendation(
        self,
        best_candidate: Dict,
        existing: Optional[Connection],
    ) -> str:
        """生成路径建议描述。"""
        if existing and existing.degree >= 0.5:
            return f'已存在直接连接 (度={existing.degree:.2f})，无需创建新路径'

        ctype = best_candidate['type']
        desc = best_candidate['description']
        degree = best_candidate['suggested_degree']

        if ctype == 'direct_new':
            return f'建议直接创建新连接，建议度={degree:.2f}。{desc}'
        elif ctype == 'via_bridge':
            return f'建议通过桥梁节点建立间接路径，路径度={degree:.2f}。{desc}'
        elif ctype == 'multi_hop':
            return f'建议使用多跳路径，路径度={degree:.2f}。{desc}'

        return desc

    def discover_potential_connections(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        min_suggested_degree: float = 0.3,
    ) -> List[Dict]:
        """
        发现网络中所有潜在的、有价值的缺失连接。

        基于:
        - 共同邻居数
        - 属性相似度
        - 连接类型兼容性
        - 传递闭包推断
        """
        conn_dict = {c.id: c for c in connections}
        self.network.build_from(things, conn_dict)

        # 使用网络图的缺失连接检测
        missing = self.network.find_missing_connections(threshold=min_suggested_degree)

        results = []
        for src_id, tgt_id, suggested in missing:
            thing_a = things.get(src_id)
            thing_b = things.get(tgt_id)

            # 计算共同邻居
            a_neighbors = set()
            b_neighbors = set()
            for conn in connections:
                if conn.source_id == src_id:
                    a_neighbors.add(conn.target_id)
                if conn.target_id == src_id:
                    a_neighbors.add(conn.source_id)
                if conn.source_id == tgt_id:
                    b_neighbors.add(conn.target_id)
                if conn.target_id == tgt_id:
                    b_neighbors.add(conn.source_id)

            common = a_neighbors & b_neighbors
            attr_sim = thing_a.similarity(thing_b) if thing_a and thing_b else 0.0

            results.append({
                'source_id': src_id,
                'source_name': thing_a.name if thing_a else src_id,
                'target_id': tgt_id,
                'target_name': thing_b.name if thing_b else tgt_id,
                'suggested_degree': suggested,
                'common_neighbors': len(common),
                'attribute_similarity': attr_sim,
                'recommendation': 'create' if suggested > 0.5 else 'consider',
            })

        results.sort(key=lambda x: x['suggested_degree'], reverse=True)
        return results

    def find_alternative_routes(
        self,
        source_id: str,
        target_id: str,
        things: Dict[str, Thing],
        connections: List[Connection],
        max_alternatives: int = 3,
    ) -> List[Dict]:
        """
        查找从 source 到 target 的所有替代路径。
        当主路径出现矛盾时，可以切换到替代路径。
        """
        conn_dict = {c.id: c for c in connections}
        self.network.build_from(things, conn_dict)

        all_paths = self.network.find_all_paths(source_id, target_id, max_depth=5)

        routes = []
        for path in all_paths[:max_alternatives]:
            # 计算路径指标
            min_degree = 1.0
            total_weight = 0.0

            for i in range(len(path) - 1):
                conn = self._find_direct_connection(path[i], path[i + 1], connections)
                if conn:
                    min_degree = min(min_degree, conn.degree)
                    total_weight += 1 - conn.degree
                else:
                    min_degree = 0
                    total_weight = 1.0

            path_names = [things.get(tid).name if things.get(tid) else tid for tid in path]

            routes.append({
                'path': path,
                'path_names': path_names,
                'hop_count': len(path) - 1,
                'min_degree': min_degree,
                'total_cost': total_weight,
                'bottleneck': min_degree < 0.3,
            })

        routes.sort(key=lambda r: r['min_degree'], reverse=True)
        return routes

    def evaluate_connection_creating_impact(
        self,
        source_id: str,
        target_id: str,
        suggested_degree: float,
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> Dict:
        """
        评估创建新连接对整个网络的影响。

        返回影响评估，包括:
        - 矛盾减少量
        - 新产生的潜在矛盾
        - 网络连通性变化
        - 中心性变化
        """
        conn_dict = {c.id: c for c in connections}
        self.network.build_from(things, conn_dict)

        # 原始网络指标
        original_components = len(self.network.connected_components())
        original_density = self.network.density()

        # 模拟添加新连接
        sim_conn = Connection(
            source_id=source_id,
            target_id=target_id,
            degree=suggested_degree,
            connection_type=ConnectionType.FUNCTIONAL,
            expected_degree=suggested_degree,
        )

        simulated_connections = connections + [sim_conn]
        simulated_dict = {c.id: c for c in simulated_connections}
        self.network.build_from(things, simulated_dict)

        # 新网络指标
        new_components = len(self.network.connected_components())
        new_density = self.network.density()

        # 计算对矛盾的影响
        source_thing = things.get(source_id)
        target_thing = things.get(target_id)

        # 新连接可能缓解的矛盾
        resolved_mismatches = 0
        for conn in connections:
            if conn.is_between(source_id, target_id) and conn.expected_degree:
                if abs(suggested_degree - conn.expected_degree) < conn.mismatch:
                    resolved_mismatches += 1

        # 新连接可能产生的矛盾
        new_potential_conflicts = 0
        for conn in connections:
            if conn.source_id == source_id or conn.target_id == source_id:
                if conn.degree > 0.7 and suggested_degree > 0.7:
                    new_potential_conflicts += 1
            if conn.source_id == target_id or conn.target_id == target_id:
                if conn.degree > 0.7 and suggested_degree > 0.7:
                    new_potential_conflicts += 1

        return {
            'source_id': source_id,
            'target_id': target_id,
            'suggested_degree': suggested_degree,
            'network_impact': {
                'components_before': original_components,
                'components_after': new_components,
                'connectivity_improved': new_components < original_components,
                'density_before': original_density,
                'density_after': new_density,
                'density_change': new_density - original_density,
            },
            'contradiction_impact': {
                'resolved_mismatches': resolved_mismatches,
                'new_potential_conflicts': new_potential_conflicts,
                'net_benefit': resolved_mismatches - new_potential_conflicts,
            },
            'recommendation': (
                'create' if resolved_mismatches > new_potential_conflicts
                else 'reject' if new_potential_conflicts > resolved_mismatches
                else 'neutral'
            ),
        }

    @property
    def discovered_paths(self) -> List[Dict]:
        return self._discovered_paths.copy()

    def __repr__(self) -> str:
        return f"<PathFinder: {len(self._discovered_paths)} paths discovered>"
