from collections import Counter
import json

def compute_depth_and_leaves(tree):
    """递归计算一棵树的深度和叶子节点数"""
    if isinstance(tree, int):
        return 1, 1
    elif isinstance(tree, list):
        max_depth = 0
        total_leaves = 0
        for subtree in tree:
            d, l = compute_depth_and_leaves(subtree)
            max_depth = max(max_depth, d)
            total_leaves += l
        return max_depth + 1, total_leaves
    else:
        raise TypeError("Unexpected tree node type")

def analyze_tree_structure(tree_structure):
    counter = Counter()
    balance = []
    for tree in tree_structure:
        depth, leaves = compute_depth_and_leaves(tree)
        counter[(depth, leaves)] += 1
        balance.append(leaves / (2**depth - 1))  # 计算平衡度
    return counter, balance

