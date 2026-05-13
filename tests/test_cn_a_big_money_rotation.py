"""测试 A 股大单净流入横截面轮动策略 v0.1.

对应设计：GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-设计.md
对应测试用例清单：同目录 -测试用例.md

测试编号 T1-T29 与测试用例清单一一对应。所有 Level 1 单测使用合成 fixture，
不连接真实 PG（参考 test_contract_db.py 用 SQLite monkey-patch 的惯例）。
"""
from __future__ import annotations


def test_smoke_package_importable():
    """M0 占位：验证策略包可被 import。后续 M1-M6 在此文件累加 T1-T29。"""
    import strategies.cn_a_big_money_rotation as pkg

    assert pkg.__version__.startswith("0.1")
