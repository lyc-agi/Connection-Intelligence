from __future__ import annotations

from typing import Dict, List

from core import Intelligence, Thing, Connection, Contradiction
from core.connection import ConnectionType
from core.contradiction import ContradictionType


def create_water_flow_scenario() -> Intelligence:
    """
    场景1: 水流场景

    矛盾: 水向低处流 (自然规律) vs 需要储水 (目的)

    智能利用规律: 用物体拦住水可以阻碍水向低处流
    解决方案: 筑坝 (创造新的连接关系)
    """
    intel = Intelligence(name="水流智能")

    # 创建事物
    water = intel.add_thing("水", {"state": "liquid", "property": "flows_downhill"})
    lowland = intel.add_thing("低处", {"height": 0})
    highland = intel.add_thing("高处", {"height": 100})
    dam = intel.add_thing("坝", {"property": "blocks_flow", "material": "concrete"})
    reservoir = intel.add_thing("水库", {"purpose": "store_water"})
    gravity = intel.add_thing("重力", {"law": "pulls_downward", "constant": 9.8})

    # 创建连接 - 自然规律
    intel.add_connection(
        source_id=gravity.id, target_id=water.id,
        degree=0.9, connection_type=ConnectionType.CAUSAL,
        weight=2.0, expected_degree=0.9
    )
    intel.add_connection(
        source_id=water.id, target_id=lowland.id,
        degree=0.8, connection_type=ConnectionType.SPATIAL,
        weight=1.5, expected_degree=0.8
    )
    intel.add_connection(
        source_id=water.id, target_id=highland.id,
        degree=0.2, connection_type=ConnectionType.SPATIAL,
        weight=1.0, expected_degree=0.2
    )

    # 创建连接 - 目的 (储水)
    intel.add_connection(
        source_id=water.id, target_id=reservoir.id,
        degree=0.1, connection_type=ConnectionType.FUNCTIONAL,
        weight=3.0, expected_degree=0.9  # 期望水流入水库
    )

    # 创建连接 - 坝的功能
    intel.add_connection(
        source_id=dam.id, target_id=water.id,
        degree=0.3, connection_type=ConnectionType.FUNCTIONAL,
        weight=2.0, expected_degree=0.8  # 期望坝能拦住水
    )

    return intel


def create_repair_scenario() -> Intelligence:
    """
    场景2: 维修场景

    矛盾: 机器故障 vs 工人收入减少
    智能: 将外部矛盾(机器故障)转换为自身矛盾(收入减少)

    这里展示 6.1 原理: 外部矛盾被转换为智能自身相关的矛盾
    """
    intel = Intelligence(name="维修智能")

    # 创建事物
    machine = intel.add_thing("机器", {"state": "broken", "function": "production"})
    worker = intel.add_thing("维修工人", {"state": "working", "skill": "repair"})
    product = intel.add_thing("产品", {"state": "needed", "value": "high"})
    money = intel.add_thing("收入", {"state": "decreasing", "need": "income"})
    factory = intel.add_thing("工厂", {"state": "operating", "goal": "production"})

    # 故障的因果链
    intel.add_connection(
        source_id=machine.id, target_id=product.id,
        degree=0.1, connection_type=ConnectionType.CAUSAL,
        weight=2.0, expected_degree=0.8
    )
    intel.add_connection(
        source_id=product.id, target_id=money.id,
        degree=0.2, connection_type=ConnectionType.CAUSAL,
        weight=2.0, expected_degree=0.7
    )

    # 工人的干预链
    intel.add_connection(
        source_id=worker.id, target_id=machine.id,
        degree=0.9, connection_type=ConnectionType.FUNCTIONAL,
        weight=3.0, expected_degree=0.8  # 工人能修理机器
    )
    intel.add_connection(
        source_id=machine.id, target_id=factory.id,
        degree=0.3, connection_type=ConnectionType.HIERARCHICAL,
        weight=1.5, expected_degree=0.7
    )
    intel.add_connection(
        source_id=factory.id, target_id=money.id,
        degree=0.2, connection_type=ConnectionType.CAUSAL,
        weight=2.0, expected_degree=0.6
    )

    # 外部约束: 工人不希望收入减少
    intel.add_external_constraint({
        'type': 'maintain_connection',
        'thing_id': money.id,
        'min_degree': 0.5,
        'description': '工人不希望收入持续减少',
    })

    return intel


def create_knowledge_extension_scenario() -> Intelligence:
    """
    场景3: 知识扩展场景

    矛盾: 已知信息 vs 未知信息
    智能: 通过已知推知未知，创造新的时空连接

    这里展示 6.3 原理: 智能创造新的连接
    """
    intel = Intelligence(name="求知智能")

    # 创建事物 - 代表不同时空的信息
    known_past = intel.add_thing("已知过去", {"time": "past", "info": "abundant", "certainty": 0.9})
    known_present = intel.add_thing("已知现在", {"time": "present", "info": "partial", "certainty": 0.7})
    unknown_future = intel.add_thing("未知未来", {"time": "future", "info": "unknown", "certainty": 0.1})
    pattern = intel.add_thing("规律", {"type": "causal", "scope": "universal"})
    prediction = intel.add_thing("预测", {"type": "inferred", "confidence": 0.0})

    # 已知之间的连接
    intel.add_connection(
        source_id=known_past.id, target_id=known_present.id,
        degree=0.9, connection_type=ConnectionType.CAUSAL,
        weight=2.0, expected_degree=0.9
    )
    intel.add_connection(
        source_id=known_past.id, target_id=pattern.id,
        degree=0.8, connection_type=ConnectionType.FUNCTIONAL,
        weight=1.5, expected_degree=0.8
    )
    intel.add_connection(
        source_id=pattern.id, target_id=known_present.id,
        degree=0.7, connection_type=ConnectionType.FUNCTIONAL,
        weight=1.5, expected_degree=0.7
    )

    # 关键矛盾: 缺少从已知到未知的连接
    intel.add_connection(
        source_id=known_present.id, target_id=unknown_future.id,
        degree=0.1, connection_type=ConnectionType.CAUSAL,
        weight=3.0, expected_degree=0.7  # 期望有强连接
    )

    # 预测的连接
    intel.add_connection(
        source_id=prediction.id, target_id=unknown_future.id,
        degree=0.2, connection_type=ConnectionType.SIMILARITY,
        weight=2.0, expected_degree=0.6
    )

    return intel


def create_purpose_conflict_scenario() -> Intelligence:
    """
    场景4: 目的冲突场景

    矛盾: 多个同等重要的目的之间的冲突
    智能: 选择影响最小的矛盾进行妥协

    这里展示 6.5 原理: 智能可以跳出矛盾所在的层次
    """
    intel = Intelligence(name="决策智能")

    # 创建事物 - 代表不同目的
    goal_a = intel.add_thing("目的A", {"priority": "high", "urgency": "medium"})
    goal_b = intel.add_thing("目的B", {"priority": "medium", "urgency": "high"})
    resource = intel.add_thing("资源", {"amount": "limited", "type": "time_money"})
    decision = intel.add_thing("决策", {"state": "pending"})
    consequence = intel.add_thing("后果", {"state": "unknown"})

    # 两个目的都需要资源
    intel.add_connection(
        source_id=resource.id, target_id=goal_a.id,
        degree=0.8, connection_type=ConnectionType.DEPENDENCY,
        weight=2.0, expected_degree=0.8
    )
    intel.add_connection(
        source_id=resource.id, target_id=goal_b.id,
        degree=0.8, connection_type=ConnectionType.DEPENDENCY,
        weight=2.0, expected_degree=0.8
    )

    # 资源有限的约束 (不能同时满足两个高需求)
    conn_a = intel.find_between(resource.id, goal_a.id)[0]
    conn_b = intel.find_between(resource.id, goal_b.id)[0]
    conn_a.add_constraint({'type': 'range', 'min': 0.0, 'max': 0.6})
    conn_b.add_constraint({'type': 'range', 'min': 0.0, 'max': 0.6})

    # 决策连接
    intel.add_connection(
        source_id=decision.id, target_id=goal_a.id,
        degree=0.5, connection_type=ConnectionType.FUNCTIONAL,
        weight=1.5, expected_degree=0.5
    )
    intel.add_connection(
        source_id=decision.id, target_id=goal_b.id,
        degree=0.5, connection_type=ConnectionType.FUNCTIONAL,
        weight=1.5, expected_degree=0.5
    )
    intel.add_connection(
        source_id=decision.id, target_id=consequence.id,
        degree=0.3, connection_type=ConnectionType.CAUSAL,
        weight=1.0, expected_degree=0.3
    )

    return intel


def run_all_scenarios() -> Dict[str, Dict]:
    """
    运行所有场景并返回结果。
    """
    results = {}

    scenarios = {
        'water_flow': ('水流场景', create_water_flow_scenario),
        'repair': ('维修场景', create_repair_scenario),
        'knowledge_extension': ('知识扩展场景', create_knowledge_extension_scenario),
        'purpose_conflict': ('目的冲突场景', create_purpose_conflict_scenario),
    }

    for key, (name, factory) in scenarios.items():
        print(f"\n{'=' * 60}")
        print(f"  运行: {name}")
        print(f"{'=' * 60}")

        intel = factory()
        print(intel.network_summary())

        # 检测矛盾
        contradictions = intel.detect_contradictions()
        print(f"\n检测到 {len(contradictions)} 个矛盾:")
        for c in contradictions:
            print(f"  - {c}")

        # 度量初始智能程度
        initial_metrics = intel.measure_intelligence()
        print(f"\n初始智能度量: D={initial_metrics['D']:.4f}")

        # 解决矛盾
        print("\n开始解决矛盾...")
        resolution_results = intel.resolve_contradictions(max_iterations=20)
        print(f"解决了 {len(resolution_results)} 个矛盾")

        # 度量解决后智能程度
        final_metrics = intel.measure_intelligence()
        print(f"最终智能度量: D={final_metrics['D']:.4f}")

        results[key] = {
            'name': name,
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'resolutions': resolution_results,
            'stats': intel.stats,
        }

    return results


if __name__ == "__main__":
    run_all_scenarios()
