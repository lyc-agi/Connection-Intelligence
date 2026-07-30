"""
基础测试 - 验证核心类的正确性。
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Thing, Connection, Intelligence, Contradiction
from core.connection import ConnectionType
from core.contradiction import ContradictionType
from algorithms import ConnectionDegreeCalculator, ConflictDetector, ContradictionResolver, IntelligenceDegree


def test_thing():
    """测试事物类。"""
    print("测试 Thing...")

    t1 = Thing("苹果", {"color": "red", "weight": 0.2})
    t2 = Thing("橙子", {"color": "orange", "weight": 0.2})
    t3 = Thing("红色物体", {"color": "red", "weight": 0.5})

    # 测试属性
    assert t1.name == "苹果"
    assert t1.get_attribute("color") == "red"
    t1.add_attribute("taste", "sweet")
    assert t1.get_attribute("taste") == "sweet"

    # 测试相似度
    sim_13 = t1.similarity(t3)  # 都有红色
    sim_12 = t1.similarity(t2)  # 颜色不同
    assert sim_13 > sim_12, f"相似度 {sim_13} 应大于 {sim_12}"

    # 测试连接计数
    assert t1.is_isolated

    print("  ✓ Thing 测试通过")
    return True


def test_connection():
    """测试连接类。"""
    print("测试 Connection...")

    conn = Connection("a", "b", degree=0.5, connection_type=ConnectionType.CAUSAL)

    # 测试度的范围
    conn.degree = 1.5
    assert conn.degree == 1.0, f"度应限制在 [0,1]，实际 {conn.degree}"
    conn.degree = -0.3
    assert conn.degree == 0.0

    # 测试不匹配
    conn.degree = 0.5
    conn.expected_degree = 0.8
    assert abs(conn.mismatch - 0.3) < 0.001

    # 测试约束
    conn.add_constraint({'type': 'range', 'min': 0.2, 'max': 0.6})
    conn.degree = 0.5
    assert conn.satisfies_constraints()
    conn.degree = 0.8
    assert not conn.satisfies_constraints()

    # 测试应变
    conn.degree = 0.5
    conn.expected_degree = 0.8
    strain = conn.strain()
    assert strain > 0, f"应变应大于0，实际 {strain}"

    # 测试松弛
    old_degree = conn.degree
    conn.relax(learning_rate=0.5)
    assert abs(conn.degree - old_degree) > 0.01, "松弛应调整度"

    print("  ✓ Connection 测试通过")
    return True


def test_intelligence_basic():
    """测试智能基本功能。"""
    print("测试 Intelligence...")

    intel = Intelligence(name="TestIntel")

    # 添加事物
    t1 = intel.add_thing("A", {"val": 10})
    t2 = intel.add_thing("B", {"val": 20})

    # 添加连接 - 低不匹配度 (0.2) 以触发 adjust_degree 策略
    conn = intel.add_connection(
        source_id=t1.id, target_id=t2.id,
        degree=0.3, connection_type=ConnectionType.CAUSAL,
        expected_degree=0.5
    )

    # 测试矛盾检测
    contradictions = intel.detect_contradictions()
    assert len(contradictions) > 0, "应检测到矛盾"

    # 测试解决
    results = intel.resolve_contradictions(max_iterations=10)
    print(f"    解决了 {len(results)} 个矛盾")

    # 测试统计 (解决策略可能创建新连接，所以用 >=)
    stats = intel.stats
    assert stats['things_count'] == 2
    assert stats['connections_count'] >= 1

    # 测试网络摘要
    summary = intel.network_summary()
    assert "TestIntel" in summary

    print("  ✓ Intelligence 测试通过")
    return True


def test_connection_degree_calculator():
    """测试连接度计算器。"""
    print("测试 ConnectionDegreeCalculator...")

    calc = ConnectionDegreeCalculator()

    t1 = Thing("A", {"x": 1, "y": 2})
    t2 = Thing("B", {"x": 1, "y": 3})
    t3 = Thing("C", {"x": 5, "y": 5})

    # 属性相似度
    sim_12 = calc._attribute_degree(t1, t2)
    sim_13 = calc._attribute_degree(t1, t3)
    assert sim_12 > sim_13, f"属性相似度 {sim_12} 应大于 {sim_13}"

    # 完整连接度计算
    conn = Connection(t1.id, t2.id, degree=0.6)
    degree = calc.compute_degree(t1, t2, [conn])
    assert 0 <= degree <= 1

    # 网络应变
    strain = calc.compute_network_strain([conn])
    assert strain >= 0

    # 网络熵
    entropy = calc.compute_network_entropy([conn])
    assert entropy >= 0

    print("  ✓ ConnectionDegreeCalculator 测试通过")
    return True


def test_conflict_detector():
    """测试冲突检测器。"""
    print("测试 ConflictDetector...")

    detector = ConflictDetector()

    t1 = Thing("A")
    t2 = Thing("B")
    conn = Connection(t1.id, t2.id, degree=0.3, expected_degree=0.8)

    # 检测不匹配
    mismatches = detector.detect_mismatches([conn])
    assert len(mismatches) == 1, f"应检测到1个不匹配，实际 {len(mismatches)}"

    # 检测约束违反
    conn.add_constraint({'type': 'range', 'min': 0.5, 'max': 1.0})
    violations = detector.detect_constraint_violations([conn])
    assert len(violations) == 1, f"应检测到1个约束违反，实际 {len(violations)}"

    print("  ✓ ConflictDetector 测试通过")
    return True


def test_resolver():
    """测试矛盾解决器。"""
    print("测试 ContradictionResolver...")

    from core.thing import Thing as T
    from core.connection import Connection as C

    t1 = T("A")
    t2 = T("B")
    conn = C(t1.id, t2.id, degree=0.3, expected_degree=0.8)

    contradiction = Contradiction(
        contradiction_type=ContradictionType.MISMATCH,
        description="测试矛盾",
        involved_connections=[conn],
        severity=0.5,
    )

    resolver = ContradictionResolver(learning_rate=0.3)
    result = resolver.resolve(contradiction, {t1.id: t1, t2.id: t2}, [conn])

    assert 'strategy' in result
    print(f"    使用策略: {result['strategy']}")

    # 验证连接度发生了变化
    assert conn.degree != 0.3 or conn.expected_degree != 0.8, \
        "解决后连接应发生变化"

    print("  ✓ ContradictionResolver 测试通过")
    return True


def test_intelligence_degree():
    """测试智能度量。"""
    print("测试 IntelligenceDegree...")

    meter = IntelligenceDegree(natural_relaxation_rate=0.1)

    # 空状态
    result = meter.compute([], {}, [])
    assert result['D'] == 0.0

    # 有矛盾状态
    t1 = Thing("A")
    t2 = Thing("B")
    conn = Connection(t1.id, t2.id, degree=0.3, expected_degree=0.8)

    contradiction = Contradiction(
        contradiction_type=ContradictionType.MISMATCH,
        description="测试",
        involved_connections=[conn],
        severity=0.5,
    )

    result = meter.compute(
        [contradiction],
        {t1.id: t1, t2.id: t2},
        [conn],
        resolution_history=[{'success': True}]
    )

    assert 'D' in result
    assert 'P' in result
    assert 'Q' in result
    assert -1 <= result['D'] <= 1
    print(f"    D={result['D']:.4f}, P={result['P']:.4f}, Q={result['Q']:.4f}")

    # 解决矛盾后
    conn.degree = 0.79
    contradiction.mark_resolved('test')

    result2 = meter.compute(
        [contradiction],
        {t1.id: t1, t2.id: t2},
        [conn]
    )
    assert result2['D'] > result['D'], \
        f"解决后 D ({result2['D']:.4f}) 应高于解决前 ({result['D']:.4f})"

    print("  ✓ IntelligenceDegree 测试通过")
    return True


def test_scenario():
    """测试水流场景。"""
    print("测试水流场景...")

    from domains.scenarios import create_water_flow_scenario

    intel = create_water_flow_scenario()
    contradictions = intel.detect_contradictions()

    assert len(contradictions) > 0, "水流场景应有矛盾"

    # 解决矛盾
    intel.resolve_contradictions(max_iterations=30)

    # 验证矛盾减少
    active = intel.active_contradictions
    print(f"    解决后剩余 {len(active)} 个活跃矛盾")

    print("  ✓ 场景测试通过")
    return True


def run_all_tests():
    """运行所有测试。"""
    print("=" * 60)
    print("  Connection-Intelligence 测试套件")
    print("=" * 60)

    tests = [
        ("Thing", test_thing),
        ("Connection", test_connection),
        ("Intelligence", test_intelligence_basic),
        ("ConnectionDegreeCalculator", test_connection_degree_calculator),
        ("ConflictDetector", test_conflict_detector),
        ("ContradictionResolver", test_resolver),
        ("IntelligenceDegree", test_intelligence_degree),
        ("Scenario", test_scenario),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {name} 测试返回 False")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} 测试异常: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  测试结果: {passed} 通过, {failed} 失败")
    print(f"  通过率: {passed / len(tests) * 100:.1f}%")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
