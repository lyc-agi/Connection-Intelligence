from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

from core.thing import Thing
from core.connection import Connection, ConnectionType
from core.contradiction import Contradiction, ContradictionType


class ContradictionResolver:
    """
    矛盾解决器 - 实现多种矛盾解决策略。

    核心思想:
    1. 矛盾是连接度的不匹配
    2. 解决矛盾 = 调整连接度/创造新连接/移除旧连接
    3. 智能在解决矛盾时也会创造新的矛盾，但新矛盾的影响更小
    4. 从更高层次审视矛盾可以找到更优解
    """

    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 100):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self._strategies: Dict[str, Callable] = {
            'gradient_descent': self._gradient_descent,
            'simulated_annealing': self._simulated_annealing,
            'graph_reorganization': self._graph_reorganization,
            'constraint_satisfaction': self._constraint_satisfaction,
            'creative_solution': self._creative_solution,
        }
        self._resolution_log: List[Dict] = []

    def resolve(
        self,
        contradiction: Contradiction,
        things: Dict[str, Thing],
        connections: List[Connection],
        strategy: Optional[str] = None,
    ) -> Dict:
        """
        解决一个矛盾。

        返回解决结果，包括采用的策略、调整的连接、成功与否。
        """
        if strategy is None:
            strategy = self._select_strategy(contradiction, len(connections))

        resolver = self._strategies.get(strategy, self._gradient_descent)
        result = resolver(contradiction, things, connections)

        self._resolution_log.append({
            'contradiction_id': contradiction.id,
            'strategy': strategy,
            'result': result,
        })

        return result

    def _select_strategy(self, contradiction: Contradiction, network_size: int) -> str:
        """
        根据矛盾特征选择最优策略。
        """
        ctype = contradiction.contradiction_type
        severity = contradiction.severity

        if ctype == ContradictionType.CIRCULAR_DEPENDENCY:
            return 'graph_reorganization'

        if ctype == ContradictionType.CONSTRAINT_VIOLATION:
            return 'constraint_satisfaction'

        if ctype == ContradictionType.CONFLICTING_DEMANDS:
            if severity > 0.6:
                return 'creative_solution'
            return 'simulated_annealing'

        if ctype == ContradictionType.MISMATCH:
            if severity < 0.3:
                return 'gradient_descent'
            return 'simulated_annealing'

        return 'gradient_descent'

    # ==================== 策略实现 ====================

    def _gradient_descent(
        self,
        contradiction: Contradiction,
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> Dict:
        """
        策略1: 梯度下降 - 沿着应变减小的方向调整连接度。

        最简单的策略，适用于轻度不匹配。
        """
        adjustments = []

        for conn in contradiction.involved_connections:
            old_degree = conn.degree
            old_strain = conn.strain()

            # 沿着期望度方向调整
            target = conn.expected_degree if conn.expected_degree is not None else conn.degree

            # 考虑约束
            for c in conn.constraints:
                if c['type'] == 'range':
                    lo = c.get('min', 0.0)
                    hi = c.get('max', 1.0)
                    target = max(lo, min(hi, target))

            # 计算梯度（简化版）
            gradient = target - conn.degree
            new_degree = conn.degree + self.learning_rate * gradient
            new_degree = max(0.0, min(1.0, new_degree))

            conn.degree = new_degree

            adjustments.append({
                'connection_id': conn.id,
                'old_degree': old_degree,
                'new_degree': new_degree,
                'old_strain': old_strain,
                'new_strain': conn.strain(),
            })

        success = all(a['new_strain'] < a['old_strain'] for a in adjustments) if adjustments else False

        if success:
            contradiction.mark_resolved('gradient_descent', {'adjustments': adjustments})

        return {
            'strategy': 'gradient_descent',
            'success': success,
            'adjustments': adjustments,
        }

    def _simulated_annealing(
        self,
        contradiction: Contradiction,
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> Dict:
        """
        策略2: 模拟退火 - 允许偶尔向更差方向移动，避免局部最优。

        适用于中度不匹配，需要跳出局部最优。
        """
        adjustments = []
        temperature = 1.0
        best_state = [(c.id, c.degree) for c in contradiction.involved_connections]
        best_strain = contradiction.total_strain

        for iteration in range(min(20, self.max_iterations)):
            # 随机选择一个连接进行调整
            if not contradiction.involved_connections:
                break

            conn = random.choice(contradiction.involved_connections)
            old_degree = conn.degree

            # 随机扰动
            perturbation = random.uniform(-0.2, 0.2) * temperature
            new_degree = max(0.0, min(1.0, old_degree + perturbation))

            conn.degree = new_degree
            current_strain = contradiction.total_strain

            # Metropolis 准则
            delta = current_strain - best_strain
            if delta < 0 or random.random() < pow(2.71828, -delta / max(temperature, 0.01)):
                if current_strain < best_strain:
                    best_strain = current_strain
                    best_state = [(c.id, c.degree) for c in contradiction.involved_connections]
            else:
                conn.degree = old_degree

            temperature *= 0.9

        # 恢复最优状态
        for cid, degree in best_state:
            conn = next((c for c in contradiction.involved_connections if c.id == cid), None)
            if conn:
                conn.degree = degree

        success = best_strain < 0.05

        if success:
            contradiction.mark_resolved('simulated_annealing', {'best_strain': best_strain})

        return {
            'strategy': 'simulated_annealing',
            'success': success,
            'best_strain': best_strain,
            'adjustments': adjustments,
        }

    def _graph_reorganization(
        self,
        contradiction: Contradiction,
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> Dict:
        """
        策略3: 图重组 - 重新组织连接网络以消除循环等结构性矛盾。

        适用于循环依赖等结构性问题。
        """
        cycle = contradiction.context.get('cycle', [])

        if len(cycle) >= 2:
            # 找到循环中最弱的连接并移除
            cycle_conns = []
            for i in range(len(cycle) - 1):
                for conn in connections:
                    if conn.source_id == cycle[i] and conn.target_id == cycle[i + 1]:
                        cycle_conns.append(conn)

            if cycle_conns:
                weakest = min(cycle_conns, key=lambda c: c.weight)
                # 降低最弱连接的度，打破循环
                old_degree = weakest.degree
                weakest.degree = max(0.1, weakest.degree * 0.3)

                contradiction.mark_resolved('graph_reorganization', {
                    'weakest_connection_id': weakest.id,
                    'old_degree': old_degree,
                    'new_degree': weakest.degree,
                    'cycle': cycle,
                })

                return {
                    'strategy': 'graph_reorganization',
                    'success': True,
                    'adjusted_connection': weakest.id,
                    'old_degree': old_degree,
                    'new_degree': weakest.degree,
                }

        return {'strategy': 'graph_reorganization', 'success': False}

    def _constraint_satisfaction(
        self,
        contradiction: Contradiction,
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> Dict:
        """
        策略4: 约束满足 - 找到满足所有约束的连接度配置。

        使用线性规划的简化版本来满足约束。
        """
        adjustments = []

        for conn in contradiction.involved_connections:
            old_degree = conn.degree

            # 找到满足所有约束的可行域
            lo, hi = 0.0, 1.0
            for c in conn.constraints:
                if c['type'] == 'range':
                    lo = max(lo, c.get('min', 0.0))
                    hi = min(hi, c.get('max', 1.0))

            # 在可行域内选择最接近期望值的值
            if conn.expected_degree is not None:
                target = max(lo, min(hi, conn.expected_degree))
            else:
                target = (lo + hi) / 2  # 如果没有期望，取中点

            conn.degree = target

            adjustments.append({
                'connection_id': conn.id,
                'old_degree': old_degree,
                'new_degree': conn.degree,
                'feasible_range': (lo, hi),
            })

        success = all(a['feasible_range'][0] <= a['new_degree'] <= a['feasible_range'][1]
                      for a in adjustments) if adjustments else False

        if success:
            contradiction.mark_resolved('constraint_satisfaction', {'adjustments': adjustments})

        return {
            'strategy': 'constraint_satisfaction',
            'success': success,
            'adjustments': adjustments,
        }

    def _creative_solution(
        self,
        contradiction: Contradiction,
        things: Dict[str, Thing],
        connections: List[Connection],
    ) -> Dict:
        """
        策略5: 创造性解决 - 创造新的连接来绕过现有矛盾。

        对应理论中的"智能创造新的连接"能力。
        这是最高级的解决策略。
        """
        affected_things = set()
        for conn in contradiction.involved_connections:
            affected_things.add(conn.source_id)
            affected_things.add(conn.target_id)

        if len(affected_things) >= 2:
            thing_list = list(affected_things)[:3]

            # 创造新的"桥梁"连接
            new_connections = []
            for i in range(len(thing_list)):
                for j in range(i + 1, len(thing_list)):
                    # 检查是否已存在
                    existing = [c for c in connections
                                if c.is_between(thing_list[i], thing_list[j])]
                    if not existing:
                        bridge_conn = Connection(
                            source_id=thing_list[i],
                            target_id=thing_list[j],
                            degree=0.7,
                            connection_type=ConnectionType.FUNCTIONAL,
                            weight=1.2,
                        )
                        bridge_conn.expected_degree = 0.7
                        new_connections.append({
                            'source': thing_list[i],
                            'target': thing_list[j],
                            'degree': 0.7,
                        })

            if new_connections:
                contradiction.mark_resolved('creative_solution', {
                    'new_connections_created': len(new_connections),
                    'details': new_connections,
                })

                return {
                    'strategy': 'creative_solution',
                    'success': True,
                    'new_connections': new_connections,
                    'message': f'创造了 {len(new_connections)} 个新连接来解决矛盾',
                }

        # 如果无法创造新连接，尝试降低矛盾严重度
        for conn in contradiction.involved_connections:
            if conn.expected_degree is not None:
                conn.expected_degree = conn.degree + (conn.expected_degree - conn.degree) * 0.5

        return {
            'strategy': 'creative_solution',
            'success': False,
            'message': '无法创造新连接，已调整期望度',
        }

    @property
    def resolution_log(self) -> List[Dict]:
        return self._resolution_log.copy()
