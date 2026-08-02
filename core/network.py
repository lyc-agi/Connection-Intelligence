from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from core.thing import Thing
from core.connection import Connection, ConnectionType


class NetworkGraph:
    """
    连接网络图 - 图论基础层。

    将事物和连接抽象为有向加权图，提供:
    - 邻接矩阵表示
    - 最短路径查找 (BFS / Dijkstra)
    - 中心性度量 (度中心性、介数中心性、接近中心性)
    - 连通分量检测
    - 环检测
    - 聚类系数

    这是智能"创造新连接"(6.3) 和"映射同构"(6.2) 的计算基础。
    """

    def __init__(self):
        self._node_index: Dict[str, int] = {}  # thing_id -> matrix index
        self._index_node: Dict[int, str] = {}  # matrix index -> thing_id
        self._adjacency: Optional[np.ndarray] = None
        self._things: Dict[str, Thing] = {}
        self._connections: Dict[str, Connection] = {}
        self._dirty = True

    def build_from(self, things: Dict[str, Thing], connections: Dict[str, Connection]) -> None:
        """从事物和连接构建图。"""
        self._things = things
        self._connections = connections
        self._dirty = True
        self._rebuild_if_dirty()

    def _rebuild_if_dirty(self) -> None:
        if not self._dirty:
            return

        nodes = list(self._things.keys())
        n = len(nodes)
        self._node_index = {tid: i for i, tid in enumerate(nodes)}
        self._index_node = {i: tid for i, tid in enumerate(nodes)}

        self._adjacency = np.zeros((n, n), dtype=np.float64)

        for conn in self._connections.values():
            src_idx = self._node_index.get(conn.source_id)
            tgt_idx = self._node_index.get(conn.target_id)
            if src_idx is not None and tgt_idx is not None:
                self._adjacency[src_idx][tgt_idx] = conn.degree
                # 无向图对称化（用于路径查找等）
                # 但保留有向信息在外部查询

        self._dirty = False

    @property
    def adjacency(self) -> np.ndarray:
        self._rebuild_if_dirty()
        return self._adjacency.copy()

    @property
    def node_count(self) -> int:
        return len(self._things)

    @property
    def edge_count(self) -> int:
        return len(self._connections)

    # ==================== 路径查找 ====================

    def shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """
        BFS 查找最短路径（忽略权重）。
        返回路径上的 thing_id 列表，或 None。
        """
        self._rebuild_if_dirty()

        if source_id not in self._node_index or target_id not in self._node_index:
            return None

        if source_id == target_id:
            return [source_id]

        visited: Set[str] = set()
        queue: deque = deque([(source_id, [source_id])])
        visited.add(source_id)

        while queue:
            current, path = queue.popleft()
            for neighbor in self._neighbors(current):
                if neighbor == target_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def shortest_path_weighted(self, source_id: str, target_id: str) -> Optional[Tuple[List[str], float]]:
        """
        Dijkstra 查找权重最短路径。
        权重 = 1 - degree（度越高，路径越短）。
        返回 (路径, 总代价) 或 None。
        """
        self._rebuild_if_dirty()

        if source_id not in self._node_index or target_id not in self._node_index:
            return None

        if source_id == target_id:
            return ([source_id], 0.0)

        # 使用优先队列的 Dijkstra
        import heapq

        dist: Dict[str, float] = {source_id: 0.0}
        prev: Dict[str, Optional[str]] = {source_id: None}
        pq: List[Tuple[float, str]] = [(0.0, source_id)]
        visited: Set[str] = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == target_id:
                # 重建路径
                path = []
                node = target_id
                while node is not None:
                    path.append(node)
                    node = prev.get(node)
                path.reverse()
                return (path, d)

            for neighbor in self._neighbors(u):
                if neighbor in visited:
                    continue
                conn = self._find_connection(u, neighbor)
                if conn is None:
                    continue
                weight = 1.0 - conn.degree  # 度越高，代价越低
                new_dist = d + weight
                if neighbor not in dist or new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = u
                    heapq.heappush(pq, (new_dist, neighbor))

        return None

    def find_all_paths(self, source_id: str, target_id: str, max_depth: int = 5) -> List[List[str]]:
        """
        查找所有简单路径（不重复访问节点），限制最大深度。
        用于"创造新连接"时探索可能的路径。
        """
        self._rebuild_if_dirty()

        if source_id not in self._node_index or target_id not in self._node_index:
            return []

        results: List[List[str]] = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) > max_depth:
                return
            if current == target_id:
                results.append(path.copy())
                return
            for neighbor in self._neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.discard(neighbor)

        dfs(source_id, [source_id], {source_id})
        return results

    def find_bridge_nodes(self, source_id: str, target_id: str) -> List[str]:
        """
        查找可以作为"桥梁"的中间节点。
        桥梁节点 = 同时连接到 source 和 target 的节点。
        这是"创造新连接"策略的核心：通过桥梁建立新路径。
        """
        self._rebuild_if_dirty()

        source_neighbors = set(self._neighbors(source_id))
        target_neighbors = set(self._neighbors(target_id))

        bridges = source_neighbors & target_neighbors
        bridges.discard(source_id)
        bridges.discard(target_id)

        return list(bridges)

    def find_missing_connections(self, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """
        查找缺失的连接 - 度低于阈值但理论上应该存在的连接。
        基于共同邻居数量推断。
        返回 [(source_id, target_id, suggested_degree), ...]
        """
        self._rebuild_if_dirty()

        missing = []
        thing_ids = list(self._things.keys())

        for i, a_id in enumerate(thing_ids):
            for b_id in thing_ids[i + 1:]:
                # 检查是否已有连接
                existing = self._find_connection(a_id, b_id)
                if existing is not None and existing.degree > threshold:
                    continue

                # 计算共同邻居数
                a_neighbors = set(self._neighbors(a_id))
                b_neighbors = set(self._neighbors(b_id))
                common = a_neighbors & b_neighbors

                if len(common) >= 2:
                    # 基于共同邻居数和属性相似度推断建议度
                    thing_a = self._things.get(a_id)
                    thing_b = self._things.get(b_id)
                    attr_sim = thing_a.similarity(thing_b) if thing_a and thing_b else 0.0
                    structural_score = min(len(common) / 5.0, 1.0)
                    suggested = 0.4 * attr_sim + 0.6 * structural_score

                    if suggested > threshold:
                        missing.append((a_id, b_id, suggested))

        return missing

    # ==================== 中心性度量 ====================

    def degree_centrality(self, thing_id: str) -> float:
        """
        度中心性 - 节点的连接数占比。
        高中心性的事物是网络的关键节点。
        """
        self._rebuild_if_dirty()

        if thing_id not in self._node_index:
            return 0.0

        n = self.node_count
        if n <= 1:
            return 0.0

        neighbors = self._neighbors(thing_id)
        return len(neighbors) / (n - 1)

    def betweenness_centrality(self, thing_id: str) -> float:
        """
        介数中心性 - 节点出现在其他节点对最短路径上的频率。
        高介数的事物是信息/影响的"瓶颈"。
        """
        self._rebuild_if_dirty()

        if thing_id not in self._node_index:
            return 0.0

        n = self.node_count
        if n <= 2:
            return 0.0

        total_pairs = 0
        on_path_count = 0

        thing_ids = list(self._things.keys())
        for i, s in enumerate(thing_ids):
            if s == thing_id:
                continue
            for t in thing_ids[i + 1:]:
                if t == thing_id:
                    continue
                path = self.shortest_path(s, t)
                if path and len(path) > 2:
                    total_pairs += 1
                    if thing_id in path[1:-1]:
                        on_path_count += 1
                elif path:
                    total_pairs += 1

        if total_pairs == 0:
            return 0.0

        return on_path_count / total_pairs

    def closeness_centrality(self, thing_id: str) -> float:
        """
        接近中心性 - 节点到所有其他节点的平均距离的倒数。
        高接近度的事物能快速影响整个网络。
        """
        self._rebuild_if_dirty()

        if thing_id not in self._node_index:
            return 0.0

        n = self.node_count
        if n <= 1:
            return 0.0

        total_dist = 0
        reachable = 0

        for other_id in self._things:
            if other_id == thing_id:
                continue
            path = self.shortest_path(thing_id, other_id)
            if path:
                total_dist += len(path) - 1
                reachable += 1

        if reachable == 0:
            return 0.0

        return reachable / (total_dist * (n - 1))

    def most_central_things(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        返回综合中心性最高的 top_k 个事物。
        """
        self._rebuild_if_dirty()

        scores: List[Tuple[str, float]] = []
        for tid in self._things:
            dc = self.degree_centrality(tid)
            bc = self.betweenness_centrality(tid)
            cc = self.closeness_centrality(tid)
            combined = 0.4 * dc + 0.4 * bc + 0.2 * cc
            scores.append((tid, combined))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ==================== 连通分量 ====================

    def connected_components(self) -> List[Set[str]]:
        """
        检测连通分量（无向图语义）。
        不同分量之间的事物完全没有连接。
        """
        self._rebuild_if_dirty()

        visited: Set[str] = set()
        components: List[Set[str]] = []

        for tid in self._things:
            if tid in visited:
                continue
            # BFS 找到整个连通分量
            component: Set[str] = set()
            queue: deque = deque([tid])
            component.add(tid)
            visited.add(tid)

            while queue:
                current = queue.popleft()
                for neighbor in self._neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

        return components

    def is_connected(self) -> bool:
        """图是否连通（单一连通分量）。"""
        components = self.connected_components()
        return len(components) <= 1

    # ==================== 环检测 ====================

    def find_cycles(self, max_length: int = 5) -> List[List[str]]:
        """
        查找图中的所有环（长度 <= max_length）。
        环代表循环依赖，是矛盾的一种来源。
        """
        self._rebuild_if_dirty()

        cycles: List[List[str]] = []
        visited: Set[str] = set()

        def dfs(start: str, current: str, path: List[str]):
            if len(path) > max_length:
                return
            for neighbor in self._neighbors(current):
                if neighbor == start and len(path) >= 2:
                    cycles.append(path.copy() + [start])
                elif neighbor not in path:
                    dfs(start, neighbor, path + [neighbor])

        for tid in self._things:
            dfs(tid, tid, [tid])

        # 去重（环的旋转和翻转视为相同）
        unique_cycles: List[List[str]] = []
        seen: Set[frozenset] = set()
        for cycle in cycles:
            key = frozenset(cycle)
            if key not in seen:
                seen.add(key)
                unique_cycles.append(cycle)

        return unique_cycles

    # ==================== 聚类系数 ====================

    def clustering_coefficient(self, thing_id: str) -> float:
        """
        计算节点的聚类系数。
        聚类系数 = 节点的邻居之间也互相连接的比例。
        高聚类系数意味着局部连接密集。
        """
        self._rebuild_if_dirty()

        neighbors = list(self._neighbors(thing_id))
        k = len(neighbors)
        if k < 2:
            return 0.0

        possible_edges = k * (k - 1) / 2
        actual_edges = 0

        for i, a in enumerate(neighbors):
            for b in neighbors[i + 1:]:
                if self._find_connection(a, b) is not None:
                    actual_edges += 1

        return actual_edges / possible_edges

    def average_clustering(self) -> float:
        """整个网络的平均聚类系数。"""
        self._rebuild_if_dirty()
        if not self._things:
            return 0.0
        return sum(self.clustering_coefficient(tid) for tid in self._things) / len(self._things)

    # ==================== 内部辅助方法 ====================

    def _neighbors(self, thing_id: str) -> List[str]:
        """获取所有邻居（出边+入边，无向语义）。"""
        result: Set[str] = set()
        for conn in self._connections.values():
            if conn.source_id == thing_id:
                result.add(conn.target_id)
            elif conn.target_id == thing_id:
                result.add(conn.source_id)
        return list(result)

    def _find_connection(self, a_id: str, b_id: str) -> Optional[Connection]:
        """查找两个节点之间的连接。"""
        for conn in self._connections.values():
            if conn.is_between(a_id, b_id):
                return conn
        return None

    # ==================== 图密度与摘要 ====================

    def density(self) -> float:
        """
        图密度 = 实际边数 / 最大可能边数。
        密度高意味着连接密集，低意味着稀疏。
        """
        n = self.node_count
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1) / 2  # 无向图
        return self.edge_count / max_edges

    def summary(self) -> str:
        """生成图的摘要描述。"""
        self._rebuild_if_dirty()
        components = self.connected_components()
        central = self.most_central_things(3)

        lines = [
            f"=== 网络图摘要 ===",
            f"节点数: {self.node_count}",
            f"边数: {self.edge_count}",
            f"图密度: {self.density():.4f}",
            f"连通分量数: {len(components)}",
            f"是否连通: {'是' if self.is_connected() else '否'}",
            f"平均聚类系数: {self.average_clustering():.4f}",
            f"环数量: {len(self.find_cycles())}",
            "",
            "最关键节点 (综合中心性):",
        ]

        for tid, score in central:
            thing = self._things.get(tid)
            name = thing.name if thing else tid
            lines.append(f"  {name}: {score:.4f}")

        return "\n".join(lines)
