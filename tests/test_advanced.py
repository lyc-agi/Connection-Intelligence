"""
深度测试 - 验证新增的图论基础层、规律库、同构映射器、分层解决器、路径搜索器。
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    Thing, Connection, ConnectionType,
    Intelligence, Contradiction, ContradictionType,
    NetworkGraph, Law, LawType, LawLibrary, IsomorphismMapper,
)
from algorithms import (
    HierarchicalResolver, ResolutionLevel,
    PathFinder, ConnectionDegreeCalculator,
)


def test_network_graph():
    """测试图论基础层。"""
    print("测试 NetworkGraph...")

    t1 = Thing("A", {"x": 1})
    t2 = Thing("B", {"x": 2})
    t3 = Thing("C", {"x": 3})
    t4 = Thing("D", {"x": 4})

    conns = {
        "c1": Connection(t1.id, t2.id, degree=0.8, connection_type=ConnectionType.CAUSAL),
        "c2": Connection(t2.id, t3.id, degree=0.6, connection_type=ConnectionType.CAUSAL),
        "c3": Connection(t3.id, t4.id, degree=0.4, connection_type=ConnectionType.FUNCTIONAL),
    }
    things = {t1.id: t1, t2.id: t2, t3.id: t3, t4.id: t4}

    graph = NetworkGraph()
    graph.build_from(things, conns)

    # 测试邻接矩阵
    adj = graph.adjacency
    assert adj.shape == (4, 4)
    assert adj[0][1] == 0.8  # A->B

    # 测试最短路径
    path = graph.shortest_path(t1.id, t4.id)
    assert path is not None
    assert len(path) == 4  # A->B->C->D

    # 测试加权最短路径
    result = graph.shortest_path_weighted(t1.id, t4.id)
    assert result is not None
    path, cost = result
    assert len(path) == 4

    # 测试桥梁节点
    bridges = graph.find_bridge_nodes(t1.id, t3.id)
    assert t2.id in bridges  # B 是 A-C 的桥梁

    # 测试中心性
    dc = graph.degree_centrality(t2.id)
    assert dc > 0  # B 有连接

    bc = graph.betweenness_centrality(t2.id)
    assert bc > 0  # B 在 A->C->D 的路径上

    # 测试连通分量
    components = graph.connected_components()
    assert len(components) == 1  # 全连通

    # 测试聚类系数
    cc = graph.clustering_coefficient(t1.id)
    assert 0 <= cc <= 1

    # 测试图密度
    d = graph.density()
    assert 0 < d < 1

    # 测试摘要
    summary = graph.summary()
    assert "网络图摘要" in summary

    print("  ✓ NetworkGraph 测试通过")
    return True


def test_law_library():
    """测试规律库。"""
    print("测试 LawLibrary...")

    library = LawLibrary()
    assert library.size > 0  # 默认注册了规律

    t1 = Thing("水", {"state": "liquid"})
    t2 = Thing("水库", {"purpose": "store"})
    conn = Connection(t1.id, t2.id, degree=0.2, expected_degree=0.9,
                      connection_type=ConnectionType.FUNCTIONAL)
    things = {t1.id: t1, t2.id: t2}
    connections = [conn]

    # 查找适用规律
    applicable = library.find_applicable(things, connections)
    assert len(applicable) > 0

    # 应用最佳规律
    result = library.apply_best(things, connections)
    assert result is not None
    assert result.get('success', False)

    # 验证连接度被调整
    assert conn.degree != 0.2  # 应该被调整了

    # 测试自定义规律
    custom_law = Law(
        name="测试规律",
        law_type=LawType.CUSTOM,
        description="测试用",
        precondition=lambda t, c, con: True,
        transformation=lambda t, c, con: {'success': True, 'modifications': []},
    )
    library.register(custom_law)
    assert library.size > 1

    print("  ✓ LawLibrary 测试通过")
    return True


def test_isomorphism_mapper():
    """测试同构映射器。"""
    print("测试 IsomorphismMapper...")

    mapper = IsomorphismMapper(similarity_threshold=0.2)

    # 创建两个结构相似的网络
    ext_a = Thing("外部A", {"type": "source", "value": 10})
    ext_b = Thing("外部B", {"type": "target", "value": 20})
    int_a = Thing("内部A", {"type": "source", "value": 12})
    int_b = Thing("内部B", {"type": "target", "value": 18})

    ext_conns = [Connection(ext_a.id, ext_b.id, degree=0.7, connection_type=ConnectionType.CAUSAL)]
    int_conns = [Connection(int_a.id, int_b.id, degree=0.5, connection_type=ConnectionType.CAUSAL)]

    ext_things = {ext_a.id: ext_a, ext_b.id: ext_b}
    int_things = {int_a.id: int_a, int_b.id: int_b}

    # 测试同构检测
    mapping = mapper.find_isomorphism(ext_things, ext_conns, int_things, int_conns)
    assert mapping is not None
    assert len(mapping) == 2

    # 测试外部矛盾转换
    ext_contradiction = Contradiction(
        contradiction_type=ContradictionType.MISMATCH,
        description="外部矛盾",
        involved_connections=ext_conns,
        severity=0.6,
    )

    internal_contra = mapper.convert_external_contradiction(
        ext_contradiction, ext_things, int_things, int_conns
    )
    assert internal_contra is not None
    assert internal_contra.severity <= ext_contradiction.severity

    # 测试结构相似度
    sim = mapper.structural_similarity(ext_things, ext_conns, int_things, int_conns)
    assert 0 <= sim <= 1

    print("  ✓ IsomorphismMapper 测试通过")
    return True


def test_hierarchical_resolver():
    """测试分层矛盾解决器。"""
    print("测试 HierarchicalResolver...")

    resolver = HierarchicalResolver()

    # 创建一个需要多层解决的矛盾
    t1 = Thing("A", {"val": 1})
    t2 = Thing("B", {"val": 2})
    conn = Connection(t1.id, t2.id, degree=0.1, expected_degree=0.9,
                      connection_type=ConnectionType.FUNCTIONAL, weight=2.0)

    contradiction = Contradiction(
        contradiction_type=ContradictionType.MISMATCH,
        description="严重不匹配",
        involved_connections=[conn],
        severity=0.8,
    )

    things = {t1.id: t1, t2.id: t2}
    connections = [conn]

    # 分层解决
    result = resolver.resolve(contradiction, things, connections)

    assert 'level' in result
    assert 'level_name' in result
    print(f"    解决层次: {result['level_name']}")

    # 验证矛盾被解决或降级
    assert result.get('success') or result.get('level', 0) >= ResolutionLevel.PURPOSE.value

    # 测试所有矛盾批量解决
    conn2 = Connection(t2.id, t1.id, degree=0.2, expected_degree=0.8,
                       connection_type=ConnectionType.DEPENDENCY)
    contra2 = Contradiction(
        contradiction_type=ContradictionType.MISMATCH,
        description="第二个矛盾",
        involved_connections=[conn2],
        severity=0.6,
    )

    results = resolver.resolve_all([contradiction, contra2], things, connections + [conn2])
    assert len(results) >= 1

    # 测试层次统计
    stats = resolver.level_statistics()
    assert isinstance(stats, dict)

    print("  ✓ HierarchicalResolver 测试通过")
    return True


def test_path_finder():
    """测试路径搜索器。"""
    print("测试 PathFinder...")

    t1 = Thing("起点", {"pos": 0})
    t2 = Thing("中间", {"pos": 5})
    t3 = Thing("终点", {"pos": 10})
    t4 = Thing("旁路", {"pos": 7})

    conns = [
        Connection(t1.id, t2.id, degree=0.8, connection_type=ConnectionType.CAUSAL),
        Connection(t2.id, t3.id, degree=0.6, connection_type=ConnectionType.CAUSAL),
        Connection(t1.id, t4.id, degree=0.4, connection_type=ConnectionType.FUNCTIONAL),
        Connection(t4.id, t3.id, degree=0.5, connection_type=ConnectionType.FUNCTIONAL),
    ]

    things = {t1.id: t1, t2.id: t2, t3.id: t3, t4.id: t4}

    finder = PathFinder()

    # 测试发现新连接路径
    route = finder.find_new_connection_route(t1.id, t3.id, things, conns)
    assert route is not None
    assert 'best_route' in route
    assert 'recommendation' in route

    # 测试发现潜在连接
    # 添加一个孤立节点
    t5 = Thing("孤立", {"pos": 3})
    things2 = {**things, t5.id: t5}
    conns2 = list(conns) + [
        Connection(t2.id, t5.id, degree=0.7),
        Connection(t5.id, t4.id, degree=0.6),
    ]

    potential = finder.discover_potential_connections(things2, conns2)
    assert isinstance(potential, list)

    # 测试替代路径
    alternatives = finder.find_alternative_routes(t1.id, t3.id, things2, conns2)
    assert len(alternatives) >= 1

    # 测试影响评估
    impact = finder.evaluate_connection_creating_impact(
        t1.id, t3.id, 0.7, things, conns
    )
    assert 'network_impact' in impact
    assert 'contradiction_impact' in impact
    assert 'recommendation' in impact

    print("  ✓ PathFinder 测试通过")
    return True


def test_intelligence_advanced():
    """测试 Intelligence 的高级功能。"""
    print("测试 Intelligence 高级功能...")

    intel = Intelligence(name="高级智能")

    # 注册智能自身
    self_thing = intel.register_self("智能自身", {"type": "agi"})
    assert intel._self_thing is not None

    # 创建事物和连接
    water = intel.add_thing("水", {"state": "liquid"})
    reservoir = intel.add_thing("水库", {"purpose": "store"})
    dam = intel.add_thing("坝", {"property": "block"})

    intel.add_connection(water.id, reservoir.id, degree=0.2,
                         connection_type=ConnectionType.FUNCTIONAL,
                         expected_degree=0.9)
    intel.add_connection(dam.id, water.id, degree=0.3,
                         connection_type=ConnectionType.FUNCTIONAL,
                         expected_degree=0.8)

    # 测试网络分析
    analysis = intel.network_analysis()
    assert "网络图摘要" in analysis

    # 测试关键节点
    critical = intel.find_critical_things()
    assert len(critical) > 0

    # 测试发现潜在连接
    potential = intel.discover_potential_connections()
    assert isinstance(potential, list)

    # 测试规律利用
    intel.detect_contradictions()
    law_result = intel.resolve_with_laws()
    assert law_result is not None

    # 测试外部矛盾转换
    ext_thing = Thing("外部问题", {"state": "broken", "type": "source"})
    ext_conn = Connection(ext_thing.id, "target", degree=0.1, expected_degree=0.8)
    ext_contradiction = Contradiction(
        contradiction_type=ContradictionType.MISMATCH,
        description="外部矛盾",
        involved_connections=[ext_conn],
        severity=0.7,
    )
    internal = intel.convert_external_contradiction(
        ext_contradiction, {ext_thing.id: ext_thing}
    )
    # 可能成功也可能失败（取决于映射），主要是验证不崩溃

    # 测试结构相似度
    ext_things = {water.id: water, reservoir.id: reservoir}
    ext_conns = [c for c in intel.connections if c.source_id == water.id or c.target_id == water.id]
    sim = intel.structural_similarity_to(ext_things, ext_conns)
    assert 0 <= sim <= 1

    # 测试连接路径查找
    route = intel.find_connection_route("水", "水库")
    assert route is not None

    # 测试统计包含新字段
    stats = intel.stats
    assert 'laws_available' in stats
    assert 'mappings_built' in stats
    assert 'has_self_representation' in stats
    assert stats['has_self_representation'] == True

    print("  ✓ Intelligence 高级功能测试通过")
    return True


def run_all_tests():
    """运行所有深度测试。"""
    print("=" * 60)
    print("  Connection-Intelligence 深度测试套件")
    print("=" * 60)

    tests = [
        ("NetworkGraph", test_network_graph),
        ("LawLibrary", test_law_library),
        ("IsomorphismMapper", test_isomorphism_mapper),
        ("HierarchicalResolver", test_hierarchical_resolver),
        ("PathFinder", test_path_finder),
        ("IntelligenceAdvanced", test_intelligence_advanced),
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
    print(f"  深度测试结果: {passed} 通过, {failed} 失败")
    print(f"  通过率: {passed / len(tests) * 100:.1f}%")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
