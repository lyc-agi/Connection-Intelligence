"""
Connection-Intelligence: 将智能视为万物之间的连接的度的协调。

基于"通用智能理论"的计算框架:
- 智能是解决矛盾的能力
- 矛盾的本质在于连接程度的不匹配
- 智能通过创造新连接、调整连接度来解决矛盾
- D = Q - P 度量智能程度
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import Intelligence, Thing, Connection
from core.connection import ConnectionType
from algorithms import ConnectionDegreeCalculator, ConflictDetector, ContradictionResolver, IntelligenceDegree
from domains.scenarios import run_all_scenarios, create_water_flow_scenario


def demo_mode():
    """演示模式 - 展示智能解决矛盾的过程。"""
    print("=" * 60)
    print("  Connection-Intelligence 演示")
    print("  智能: 万物之间的连接度的协调")
    print("=" * 60)

    # 创建智能
    intel = Intelligence(name="演示智能", learning_rate=0.15)

    # 创建一个简单的连接网络: 水 -> 水库 (矛盾: 期望度高, 实际度低)
    water = intel.add_thing("水", {"state": "liquid"})
    reservoir = intel.add_thing("水库", {"purpose": "store_water"})
    dam = intel.add_thing("坝", {"property": "blocks_flow"})
    gravity = intel.add_thing("重力", {"law": "downward"})

    # 自然连接 (水往低处流)
    intel.add_connection(
        source_id=gravity.id, target_id=water.id,
        degree=0.9, connection_type=ConnectionType.CAUSAL,
        expected_degree=0.9, weight=2.0
    )

    # 矛盾: 水流入水库的度太低
    intel.add_connection(
        source_id=water.id, target_id=reservoir.id,
        degree=0.2, connection_type=ConnectionType.FUNCTIONAL,
        expected_degree=0.9, weight=3.0
    )

    # 坝的功能连接 (已存在但度不足)
    intel.add_connection(
        source_id=dam.id, target_id=water.id,
        degree=0.3, connection_type=ConnectionType.FUNCTIONAL,
        expected_degree=0.8, weight=2.0
    )

    print("\n初始网络:")
    print(intel.network_summary())

    # 检测矛盾
    contradictions = intel.detect_contradictions()
    print(f"\n检测到 {len(contradictions)} 个矛盾:")
    for c in contradictions:
        print(f"  - {c}")

    # 初始度量
    metrics_before = intel.measure_intelligence()
    print(f"\n初始智能度量: D={metrics_before['D']:.4f}")
    print(f"  P (自然解决概率)={metrics_before['P']:.4f}")
    print(f"  Q (智能解决概率)={metrics_before['Q']:.4f}")

    # 解决矛盾
    print("\n开始解决矛盾...")
    results = intel.resolve_contradictions(max_iterations=15)
    print(f"解决了 {len(results)} 个矛盾")
    for r in results:
        print(f"  - 策略: {r.get('strategy', 'unknown')}, 成功: {r.get('success', False)}")

    # 解决后度量
    metrics_after = intel.measure_intelligence()
    print(f"\n最终智能度量: D={metrics_after['D']:.4f}")
    print(f"  P (自然解决概率)={metrics_after['P']:.4f}")
    print(f"  Q (智能解决概率)={metrics_after['Q']:.4f}")

    # 显示最终网络
    print("\n最终网络:")
    print(intel.network_summary())

    # 分析改善
    print(f"\n智能改善: ΔD={metrics_after['D'] - metrics_before['D']:.4f}")
    print(f"活跃矛盾: {len(intel.active_contradictions)}")


def interactive_mode():
    """交互模式。"""
    intel = Intelligence(name="交互智能", learning_rate=0.15)

    print("\n" + "=" * 60)
    print("  Connection-Intelligence 交互模式")
    print("=" * 60)
    print("  命令:")
    print("    create <name> [attrs]  - 创建事物")
    print("    connect <src> <tgt> [degree] [type] - 创建连接")
    print("    expect <conn_id> <degree> - 设置期望度")
    print("    detect                 - 检测矛盾")
    print("    resolve                - 解决矛盾")
    print("    measure                - 度量智能程度")
    print("    status                 - 查看状态")
    print("    network                - 显示网络")
    print("    quit                   - 退出")
    print("=" * 60)

    thing_map = {}  # name -> thing_id

    while True:
        try:
            cmd = input("\n> ").strip()
            if not cmd:
                continue

            parts = cmd.split()
            action = parts[0].lower()

            if action in ['quit', 'exit', 'q']:
                print("再见！")
                break

            elif action == 'create':
                if len(parts) < 2:
                    print("用法: create <name> [key=val ...]")
                    continue
                name = parts[1]
                attrs = {}
                for arg in parts[2:]:
                    if '=' in arg:
                        k, v = arg.split('=', 1)
                        try:
                            v = int(v)
                        except ValueError:
                            try:
                                v = float(v)
                            except ValueError:
                                pass
                        attrs[k] = v
                thing = intel.add_thing(name, attrs)
                thing_map[name] = thing.id
                print(f"创建事物: {name} (id={thing.id})")

            elif action == 'connect':
                if len(parts) < 3:
                    print("用法: connect <src_name> <tgt_name> [degree] [type]")
                    continue
                src_name = parts[1]
                tgt_name = parts[2]
                degree = float(parts[3]) if len(parts) > 3 else 0.5
                conn_type = ConnectionType.CUSTOM
                if len(parts) > 4:
                    type_map = {t.name.lower(): t for t in ConnectionType}
                    conn_type = type_map.get(parts[4].lower(), ConnectionType.CUSTOM)

                src_id = thing_map.get(src_name)
                tgt_id = thing_map.get(tgt_name)
                if not src_id or not tgt_id:
                    print(f"错误: 未知事物名称。可用: {list(thing_map.keys())}")
                    continue

                conn = intel.add_connection(
                    source_id=src_id, target_id=tgt_id,
                    degree=degree, connection_type=conn_type
                )
                if conn:
                    print(f"创建连接: {src_name} --[{degree}]--> {tgt_name} (id={conn.id})")
                else:
                    print("错误: 创建连接失败")

            elif action == 'expect':
                if len(parts) < 3:
                    print("用法: expect <conn_id> <degree>")
                    continue
                conn_id = parts[1]
                degree = float(parts[2])
                conn = intel.get_connection(conn_id)
                if conn:
                    conn.expected_degree = degree
                    print(f"设置连接 {conn_id} 期望度为 {degree}")
                else:
                    print(f"错误: 未找到连接 {conn_id}")

            elif action == 'detect':
                contradictions = intel.detect_contradictions()
                print(f"\n检测到 {len(contradictions)} 个矛盾:")
                for c in contradictions:
                    print(f"  - {c}")

            elif action == 'resolve':
                results = intel.resolve_contradictions()
                print(f"\n解决了 {len(results)} 个矛盾")
                active = intel.active_contradictions
                print(f"剩余 {len(active)} 个活跃矛盾")

            elif action == 'measure':
                metrics = intel.measure_intelligence()
                print(f"\n智能度量:")
                print(f"  D = {metrics['D']:.4f}")
                print(f"  P = {metrics['P']:.4f} (自然解决概率)")
                print(f"  Q = {metrics['Q']:.4f} (智能解决概率)")
                for k, v in metrics.get('details', {}).items():
                    print(f"  {k} = {v}")

            elif action == 'status':
                stats = intel.stats
                print(f"\n状态:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")

            elif action == 'network':
                print(intel.network_summary())

            else:
                print(f"未知命令: {action}")

        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Connection-Intelligence: 将智能视为万物之间的连接的度的协调"
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['demo', 'interactive', 'scenarios', 'test'],
        default='demo',
        help='运行模式',
    )
    parser.add_argument(
        '--scenario', '-s',
        choices=['water_flow', 'repair', 'knowledge_extension', 'purpose_conflict'],
        default=None,
        help='运行特定场景',
    )

    args = parser.parse_args()

    if args.mode == 'demo':
        demo_mode()
    elif args.mode == 'interactive':
        interactive_mode()
    elif args.mode == 'scenarios':
        if args.scenario:
            factories = {
                'water_flow': create_water_flow_scenario,
            }
            factory = factories.get(args.scenario)
            if factory:
                intel = factory()
                print(intel.network_summary())
                intel.detect_contradictions()
                intel.resolve_contradictions()
                print(intel.network_summary())
            else:
                print(f"场景 {args.scenario} 尚未实现")
        else:
            run_all_scenarios()
    elif args.mode == 'test':
        from tests.test_basic import run_all_tests
        success = run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
