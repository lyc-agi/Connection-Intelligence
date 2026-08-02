from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.thing import Thing
from core.connection import Connection, ConnectionType
from core.contradiction import Contradiction, ContradictionType


class LawType(Enum):
    """规律类型。"""
    NATURAL = "natural"           # 自然规律 (如水往低处流)
    CAUSAL = "causal"             # 因果规律 (如A导致B)
    STRUCTURAL = "structural"     # 结构规律 (如层次包含)
    TRANSFORMATION = "transformation"  # 变换规律 (如加坝可阻水)
    CONSERVATION = "conservation" # 守恒规律 (如能量守恒)
    EMERGENT = "emergent"         # 涌现规律 (如整体大于部分)
    CUSTOM = "custom"             # 自定义规律


class Law:
    """
    规律 - 智能利用的可复用模式。

    理论 6.4: 智能发挥作用时利用了若干种规律。
    矛盾之所以出现，是因为只靠"自然规律"无法实现一些目的。
    智能利用规律来变换连接，达到目的。

    规律的要素:
    - 条件 (precondition): 何时适用此规律
    - 变换 (transformation): 如何改变连接网络
    - 效果 (effect): 变换后预期的结果
    - 置信度 (confidence): 规律的可靠性 [0,1]
    """

    def __init__(
        self,
        name: str,
        law_type: LawType,
        description: str,
        precondition: Optional[Callable] = None,
        transformation: Optional[Callable] = None,
        confidence: float = 1.0,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.law_type = law_type
        self.description = description
        self.precondition = precondition
        self.transformation = transformation
        self.confidence = max(0.0, min(1.0, confidence))
        self.application_count = 0
        self.success_count = 0

    def applies_to(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction] = None,
    ) -> bool:
        """
        检查此规律是否适用于当前状态。
        """
        if self.precondition is None:
            return True
        return self.precondition(things, connections, contradiction)

    def apply(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction] = None,
    ) -> Dict[str, Any]:
        """
        应用此规律，变换连接网络。

        返回变换结果，包括:
        - modifications: 对现有连接的修改
        - new_connections: 建议创建的新连接
        - removed_connections: 建议移除的连接
        - success: 是否成功应用
        """
        self.application_count += 1

        if self.transformation is None:
            return {
                'success': False,
                'message': '规律没有定义变换函数',
            }

        result = self.transformation(things, connections, contradiction)
        result.setdefault('success', True)
        result.setdefault('law_name', self.name)
        result.setdefault('law_type', self.law_type.value)

        if result.get('success', False):
            self.success_count += 1

        return result

    @property
    def reliability(self) -> float:
        """规律的可靠性 = 成功应用次数 / 总应用次数。"""
        if self.application_count == 0:
            return self.confidence
        return self.success_count / self.application_count

    def __repr__(self) -> str:
        return (f"<Law: {self.name} [{self.law_type.value}] "
                f"confidence={self.confidence:.2f} "
                f"applied={self.application_count}>")


class LawLibrary:
    """
    规律库 - 管理智能可用的所有规律。

    智能从规律库中选择适用的规律来解决矛盾。
    """

    def __init__(self):
        self._laws: Dict[str, Law] = {}
        self._register_default_laws()

    def _register_default_laws(self) -> None:
        """注册默认的自然规律集。"""
        # 规律1: 阻断规律 - 通过插入中间事物来减弱连接度
        self.register(Law(
            name="阻断规律",
            law_type=LawType.TRANSFORMATION,
            description="在两个事物之间插入一个阻断事物，可以减弱它们之间的连接度",
            precondition=self._precondition_has_mismatch,
            transformation=self._transform_block_flow,
            confidence=0.8,
        ))

        # 规律2: 桥接规律 - 通过中间事物建立间接连接
        self.register(Law(
            name="桥接规律",
            law_type=LawType.TRANSFORMATION,
            description="通过共同邻居建立两个事物之间的间接连接",
            precondition=self._precondition_has_gap,
            transformation=self._transform_bridge,
            confidence=0.7,
        ))

        # 规律3: 因果传递规律 - A->B 且 B->C 则 A 间接影响 C
        self.register(Law(
            name="因果传递规律",
            law_type=LawType.CAUSAL,
            description="因果连接具有传递性：A->B 且 B->C 蕴含 A->C",
            precondition=self._precondition_causal_chain,
            transformation=self._transform_causal_transitivity,
            confidence=0.9,
        ))

        # 规律4: 对称互补规律 - 增强一方的连接度可补偿另一方的不足
        self.register(Law(
            name="互补规律",
            law_type=LawType.CONSERVATION,
            description="当资源有限时，增强关键路径的连接度可补偿非关键路径的不足",
            precondition=self._precondition_resource_conflict,
            transformation=self._transform_complement,
            confidence=0.6,
        ))

        # 规律5: 层次提升规律 - 将矛盾提升到更高抽象层次
        self.register(Law(
            name="层次提升规律",
            law_type=LawType.STRUCTURAL,
            description="将具体矛盾提升到更高抽象层次，通过调整目的来缓解矛盾",
            precondition=self._precondition_multiple_conflicts,
            transformation=self._transform_elevate,
            confidence=0.5,
        ))

    def register(self, law: Law) -> None:
        self._laws[law.id] = law

    def get(self, law_id: str) -> Optional[Law]:
        return self._laws.get(law_id)

    def find_applicable(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction] = None,
    ) -> List[Law]:
        """
        查找当前状态下所有适用的规律。
        按可靠性排序。
        """
        applicable = [
            law for law in self._laws.values()
            if law.applies_to(things, connections, contradiction)
        ]
        applicable.sort(key=lambda l: l.reliability, reverse=True)
        return applicable

    def apply_best(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        选择并应用最可靠且适用的规律。
        """
        applicable = self.find_applicable(things, connections, contradiction)
        if not applicable:
            return None

        best = applicable[0]
        return best.apply(things, connections, contradiction)

    @property
    def laws(self) -> List[Law]:
        return list(self._laws.values())

    @property
    def size(self) -> int:
        return len(self._laws)

    # ==================== 默认规律的前置条件和变换 ====================

    def _precondition_has_mismatch(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction],
    ) -> bool:
        """检查是否存在度不匹配（实际 > 期望，需要减弱）。"""
        for conn in connections:
            if conn.expected_degree is not None and conn.degree > conn.expected_degree + 0.1:
                return True
        return False

    def _transform_block_flow(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction],
    ) -> Dict[str, Any]:
        """阻断变换：降低过强连接的度。"""
        modifications = []
        for conn in connections:
            if conn.expected_degree is not None and conn.degree > conn.expected_degree + 0.1:
                old_degree = conn.degree
                # 逐步降低到期望值
                conn.degree = conn.degree - (conn.degree - conn.expected_degree) * 0.5
                modifications.append({
                    'connection_id': conn.id,
                    'old_degree': old_degree,
                    'new_degree': conn.degree,
                    'action': 'reduce',
                })

        return {
            'success': len(modifications) > 0,
            'modifications': modifications,
            'message': f'阻断了 {len(modifications)} 个过强连接',
        }

    def _precondition_has_gap(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction],
    ) -> bool:
        """检查是否存在连接缺失（实际 < 期望，需要增强）。"""
        for conn in connections:
            if conn.expected_degree is not None and conn.degree < conn.expected_degree - 0.1:
                return True
        return False

    def _transform_bridge(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction],
    ) -> Dict[str, Any]:
        """桥接变换：增强过弱连接的度。"""
        modifications = []
        for conn in connections:
            if conn.expected_degree is not None and conn.degree < conn.expected_degree - 0.1:
                old_degree = conn.degree
                # 逐步增强到期望值
                conn.degree = conn.degree + (conn.expected_degree - conn.degree) * 0.5
                modifications.append({
                    'connection_id': conn.id,
                    'old_degree': old_degree,
                    'new_degree': conn.degree,
                    'action': 'strengthen',
                })

        return {
            'success': len(modifications) > 0,
            'modifications': modifications,
            'message': f'桥接了 {len(modifications)} 个过弱连接',
        }

    def _precondition_causal_chain(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction],
    ) -> bool:
        """检查是否存在因果链 A->B->C 但缺少 A->C。"""
        causal_conns = [c for c in connections if c.connection_type == ConnectionType.CAUSAL]
        conn_pairs = {(c.source_id, c.target_id) for c in causal_conns}

        for c1 in causal_conns:
            for c2 in causal_conns:
                if c1.target_id == c2.source_id and c1.source_id != c2.target_id:
                    if (c1.source_id, c2.target_id) not in conn_pairs:
                        return True
        return False

    def _transform_causal_transitivity(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction],
    ) -> Dict[str, Any]:
        """因果传递变换：为因果链补全传递连接。"""
        causal_conns = [c for c in connections if c.connection_type == ConnectionType.CAUSAL]
        conn_pairs = {(c.source_id, c.target_id): c for c in causal_conns}

        new_connections = []
        for c1 in causal_conns:
            for c2 in causal_conns:
                if c1.target_id == c2.source_id and c1.source_id != c2.target_id:
                    pair = (c1.source_id, c2.target_id)
                    if pair not in conn_pairs:
                        # 建议创建传递连接
                        trans_degree = min(c1.degree, c2.degree) * 0.8
                        new_connections.append({
                            'source': c1.source_id,
                            'target': c2.target_id,
                            'degree': trans_degree,
                            'type': ConnectionType.CAUSAL,
                            'reason': f'因果传递: {c1.source_id}->{c1.target_id}->{c2.target_id}',
                        })

        return {
            'success': len(new_connections) > 0,
            'new_connections': new_connections,
            'message': f'发现 {len(new_connections)} 个可补全的因果传递连接',
        }

    def _precondition_resource_conflict(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction],
    ) -> bool:
        """检查是否存在资源冲突。"""
        if contradiction and contradiction.contradiction_type == ContradictionType.CONFLICTING_DEMANDS:
            return True

        # 检查是否有节点有多个高连接度输出
        out_degrees: Dict[str, List[float]] = {}
        for conn in connections:
            out_degrees.setdefault(conn.source_id, []).append(conn.degree)

        for degrees in out_degrees.values():
            if len(degrees) >= 2 and all(d > 0.5 for d in degrees):
                return True
        return False

    def _transform_complement(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction],
    ) -> Dict[str, Any]:
        """互补变换：保持总连接度守恒，重新分配。"""
        modifications = []

        # 找出冲突的连接组
        out_conns: Dict[str, List[Connection]] = {}
        for conn in connections:
            out_conns.setdefault(conn.source_id, []).append(conn)

        for source_id, conns in out_conns.items():
            if len(conns) >= 2:
                # 按权重排序，优先保证高权重连接
                conns_sorted = sorted(conns, key=lambda c: c.weight, reverse=True)
                total = sum(c.degree for c in conns_sorted)

                if total > 1.0:  # 总和超过1，需要重新分配
                    for i, conn in enumerate(conns_sorted):
                        old_degree = conn.degree
                        # 高权重的保留更多
                        ratio = conn.weight / sum(c.weight for c in conns_sorted)
                        conn.degree = min(1.0, total * ratio)
                        if abs(conn.degree - old_degree) > 0.01:
                            modifications.append({
                                'connection_id': conn.id,
                                'old_degree': old_degree,
                                'new_degree': conn.degree,
                                'action': 'rebalance',
                            })

        return {
            'success': len(modifications) > 0,
            'modifications': modifications,
            'message': f'互补重分配了 {len(modifications)} 个连接',
        }

    def _precondition_multiple_conflicts(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction],
    ) -> bool:
        """检查是否存在多个矛盾（适合层次提升）。"""
        if contradiction is None:
            return False
        return contradiction.severity > 0.5

    def _transform_elevate(
        self,
        things: Dict[str, Thing],
        connections: List[Connection],
        contradiction: Optional[Contradiction],
    ) -> Dict[str, Any]:
        """层次提升变换：调整期望值而非实际值。"""
        modifications = []

        if contradiction:
            for conn in contradiction.involved_connections:
                if conn.expected_degree is not None:
                    old_expected = conn.expected_degree
                    # 将期望向实际靠拢（妥协）
                    conn.expected_degree = conn.degree + (conn.expected_degree - conn.degree) * 0.3
                    modifications.append({
                        'connection_id': conn.id,
                        'old_expected': old_expected,
                        'new_expected': conn.expected_degree,
                        'action': 'elevate_expectation',
                    })

        return {
            'success': len(modifications) > 0,
            'modifications': modifications,
            'message': f'层次提升调整了 {len(modifications)} 个期望',
        }

    def __repr__(self) -> str:
        return f"<LawLibrary: {len(self._laws)} laws>"
