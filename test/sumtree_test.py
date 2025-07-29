import numpy as np
import pytest
from rl_agents.utils.sumtree import SumTree      # adaptez si besoin

tolerance = 1E-6

# ---------- état initial ----------
def test_initial_state():
    tree = SumTree(4)
    assert tree.layer_lens == [4, 2, 1]
    assert tree.sum() < tolerance

# ---------- __setitem__ : mise à jour simple ----------
def test_setitem_single():
    tree = SumTree(4)
    tree[0] = 5.0
    assert tree[0] == 5.0
    # propagation jusqu’à la racine
    assert tree.node_layers[1][0] == 5.0
    assert tree.node_layers[2][0] == 5.0
    assert abs(tree.sum() - 5.0) < tolerance

# ---------- __setitem__ : assignation vectorisée ----------
def test_setitem_batch():
    tree = SumTree(4)
    tree[[0, 1]] = [5.0, 3.0]
    assert np.allclose(tree.node_layers[0][:2], [5.0, 3.0])
    assert abs(tree.sum() - 8.0) < tolerance

# ---------- add() : règle des poids, count, sum ----------
def test_add_updates_count_and_sum():
    tree = SumTree(4)
    tree.add(0) # poids = 0 (premier élément)
    assert tree[0] < tolerance
    assert tree.sum() < tolerance

# ---------- sample : priorité aux nouvelles feuilles ----------
def test_sample_prefers_new_leafs_first():
    tree = SumTree(4)
    tree.add(3)             # new_leafs = [3]
    out = tree.sample(1)
    assert out == [3]
    assert tree.new_leafs == []

# ---------- sample : tirage pondéré, résultat déterministe ----------
def test_sample_deterministic(monkeypatch):
    tree = SumTree(2)
    tree[0] = 1.0
    tree[1] = 1.0
    tree.new_leafs = []     # vider la file « fresh »

    # forcer rand() à 0 → cible la feuille 0
    monkeypatch.setattr(np.random, "rand", lambda n: np.zeros(n))
    out = tree.sample(1)
    assert out == [0]