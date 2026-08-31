import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from interval_pattern import Segment, Object
from local_neighborhood import gen_neighboors, get_dist, get_weights


random_seed = 1312562
random.seed(random_seed)


data = load_breast_cancer()

x = data.data
y = data.target

feature_names = data.feature_names
target_names = data.target_names


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=random_seed,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=random_seed
)

model.fit(x_train, y_train)


train_vals = model.predict(x_train)
test_vals = model.predict(x_test)


print("Train accuracy:", accuracy_score(y_train, train_vals))
print("Test accuracy:", accuracy_score(y_test, test_vals))

print("Confusion matrix:")
print(confusion_matrix(y_test, test_vals))


def f(obj):
    vals = np.array([obj.features])
    ans = model.predict(vals)
    return ans[0]


class Node:
    def __init__(
        self,
        ind=-1,
        lb=0,
        rb=0,
        left=None,
        right=None,
        predict=None
    ):
        self.ind = ind
        self.segment = Segment(lb, rb)
        self.left = left
        self.right = right
        self.predict = predict

    def is_leaf(self):
        return self.left is None and self.right is None

    def goes_left(self, o):
        return self.segment.is_point_in(
            o.features[self.ind]
        )

    def __repr__(self):
        if self.is_leaf():
            return f"Leaf(class={self.predict})"
        return (
            f"Node("
            f"feature={self.ind}, "
            f"segment={self.segment}"
            f")"
        )


class DecisionTree:
    def __init__(self, root=None):
        self.root = root


    def predict(self, o):
        leaf = self.get_leaf(o)
        return leaf.predict


    def get_path(self, o):
        ans = []
        cur = self.root
        while not cur.is_leaf():
            ans.append(cur)
            if cur.goes_left(o):
                cur = cur.left
            else:
                cur = cur.right
        ans.append(cur)
        return ans


    def get_leaf(self, o):
        return self.get_path(o)[-1]



def contains_object(tree, o, z):
    l1 = tree.get_leaf(o)
    l2 = tree.get_leaf(z)
    return l1 is l2


def extent(tree, o, local_samples):
    ans = []
    for z in local_samples:
        if contains_object(tree, o, z):
            ans.append(z)
    return ans


def calc_purity(tree, local_samples, local_predictions, weights, o):
    ind = 0
    s = 0
    our_f = f(o)
    have = 0
    for z in local_samples:
        if not contains_object(tree, o, z):
            ind += 1
            continue
        s += weights[ind]
        if local_predictions[ind] == our_f:
            have += weights[ind]
        ind += 1
    if s == 0:
        return 0
    return have / s


def is_more_general(tree1, tree2, o, local_samples):
    for z in local_samples:
        if contains_object(tree2, o, z):
            if not contains_object(tree1, o, z):
                return False

    return True

def build_candidate_tree(o, our_depth, max_depth, taken_inds, local_samples):
    if our_depth >= max_depth:
        empt = Node()
        empt.predict = random.randint(0, 1)
        return empt
    ind = random.randint(0, len(o.features) - 1)
    while ind in taken_inds:
        ind = random.randint(0, len(o.features) - 1)
    bounds = []
    for obj in local_samples:
        bounds.append(obj.features[ind])
    bounds.sort()
    sz = len(bounds)
    lb = bounds[random.randint(0, sz - 1)]
    rb = bounds[random.randint(0, sz - 1)]
    if lb > rb:
        lb, rb = rb, lb
    nxt_taken_inds = taken_inds.copy()
    nxt_taken_inds.append(ind)
    left = build_candidate_tree(o, our_depth + 1, max_depth, nxt_taken_inds, local_samples)
    right = build_candidate_tree(o, our_depth + 1, max_depth, nxt_taken_inds, local_samples)
    root = Node(left=left, right=right, lb=lb, rb=rb, ind=ind, )
    return root

def get_random_C(n, k_range):
    k = random.randint(1, k_range)
    inds = []
    for iter in range(k):
        ind = random.randint(0, n - 1)
        while ind in inds:
            ind = random.randint(0, n - 1)
        inds.append(ind)
    inds.sort()
    return inds        

def build_candidate_trees(o, local_samples):
    cnt = 50
    ans = []
    n = len(local_samples)
    k = 30
    max_depth = 5
    for iter in range(cnt):
        inds = get_random_C(n, k)
        nxt_local_samples = [o]
        for ind in inds:
            nxt_local_samples.append(local_samples[ind])
        root = build_candidate_tree(o, 0, max_depth, [], nxt_local_samples)
        DT = DecisionTree(root)
        ans.append(DT)
    return ans
    

def build_local_generalization_graph(candidate_trees, local_samples, local_predictions, weights, o):
    g = nx.DiGraph()
    N = len(candidate_trees)
    for i in range(N):
        p = candidate_trees[i]
        a = extent(p, o, local_samples)
        support = len(a)
        purity = calc_purity(p, local_samples, local_predictions, weights, o)
        coverage = support / len(local_samples)

        description = p.get_path(o)

        g.add_node(
            i,
            tree=p,
            support=support,
            coverage=coverage,
            purity=purity,
            predict=f(o),
            description=description
        )

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            p_spec = candidate_trees[i]
            p_gen = candidate_trees[j]
            if not is_more_general(p_gen, p_spec, o, local_samples):
                continue
            if is_more_general(p_spec, p_gen, o, local_samples):
                continue
            direct = True
            for k in range(N):
                if k == i or k == j:
                    continue
                p_check = candidate_trees[k]
                v1 = is_more_general(p_check, p_spec, o, local_samples) and not is_more_general(p_spec, p_check, o, local_samples)
                v2 = is_more_general(p_gen, p_check, o, local_samples) and not is_more_general(p_check, p_gen, o, local_samples)
                if v1 and v2:
                    direct = False
                    break
            if direct:
                g.add_edge(i, j)
    return g


def visualize_local_generalization_graph(g, explanation=None):
    if len(g.nodes) == 0:
        print("No graph")
        return
    gens = list(nx.topological_generations(g))
    coords = {}
    for y in range(len(gens)):
        nodes = list(gens[y])
        for j in range(len(nodes)):
            x = j - (len(nodes) - 1) / 2
            coords[nodes[j]] = (x * 2.5, y * 2)
    labels = {}
    for node in g.nodes:
        data = g.nodes[node]
        labels[node] = (
            f"P{node}\n"
            f"support: {data['support']}\n"
            f"purity: {data['purity']:.2f}"
        )

    purities = []
    lens = []

    for node in g.nodes:
        purities.append(g.nodes[node]["purity"])
        lens.append(1800 + g.nodes[node]["support"] * 5)

    plt.figure(figsize=(16, 10))

    nodes = nx.draw_networkx_nodes(
        g,
        coords,
        node_size=lens,
        node_color=purities,
        cmap=plt.cm.viridis,
        vmin=0,
        vmax=1,
        edgecolors="black",
        linewidths=1.5
    )

    nx.draw_networkx_edges(
        g,
        coords,
        arrows=True,
        arrowsize=22,
        width=1.5,
        connectionstyle="arc3,rad=0.05"
    )

    nx.draw_networkx_labels(
        g,
        coords,
        labels=labels,
        font_size=9,
        font_weight="bold"
    )

    if explanation is not None:
        nx.draw_networkx_nodes(
            g,
            coords,
            nodelist=[explanation],
            node_size=[lens[list(g.nodes).index(explanation)] + 500],
            node_color=[g.nodes[explanation]["purity"]],
            cmap=plt.cm.viridis,
            vmin=0,
            vmax=1,
            edgecolors="red",
            linewidths=4
        )

    plt.colorbar(nodes, label="Purity")

    plt.title(
        "Local Generalization Graph\n"
        "Specific patterns → More general patterns",
        fontsize=16
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()

need_purity = 0.2
need_support = 10

def get_final_explanation(g, need_purity, need_support):
    take = []
    for node in g.nodes:
        purity = g.nodes[node]["purity"]
        support = g.nodes[node]["support"]
        if purity >= need_purity and support >= need_support:
            take.append(node)
    mx_depth_nodes = []

    for node in take:
        have_nxt = False
        for other in take:
            if node == other:
                continue
            if nx.has_path(g, node, other):
                have_nxt = True
                break
        if not have_nxt:
            mx_depth_nodes.append(node)

    if len(mx_depth_nodes) == 0:
        return None
    best = mx_depth_nodes[0]

    for node in mx_depth_nodes:
        if g.nodes[node]["purity"] > g.nodes[best]["purity"]:
            best = node
        elif g.nodes[node]["purity"] == g.nodes[best]["purity"]:
            if g.nodes[node]["support"] > g.nodes[best]["support"]:
                best = node

    return best


feature_std = np.std(x_train, axis=0)

test_ind = 141
o = Object(x_train[test_ind].tolist())

local_samples = gen_neighboors(o, feature_std)
dist = get_dist(o, local_samples, feature_std)
order = np.argsort(dist)
local_samples = [
    local_samples[i]
    for i in order
]
dist = dist[order]
local_predictions = []

for obj in local_samples:
    local_predictions.append(model.predict([obj.features])[0])

weights = get_weights(dist, 2)

candidate_trees = build_candidate_trees(o, local_samples)



g = build_local_generalization_graph(candidate_trees, local_samples, local_predictions, weights, o)
explanation = get_final_explanation(g, need_purity, need_support)
def print_tree(node, depth=0):
    if node.is_leaf():
        print("    " * depth + f"Leaf: class = {node.predict}")
        return

    print(
        "    " * depth +
        f"feature {node.ind}: [{node.segment.l}, {node.segment.r}]"
    )

    print("    " * depth + "Left:")
    print_tree(node.left, depth + 1)

    print("    " * depth + "Right:")
    print_tree(node.right, depth + 1)


def print_explanation(explanation):
    if explanation is None:
        print("No explanation")
        return

    print("Node:", explanation)
    print("Support:", g.nodes[explanation]["support"])
    print("Coverage:", g.nodes[explanation]["coverage"])
    print("Purity:", g.nodes[explanation]["purity"])
    print("Predicted class:", g.nodes[explanation]["predict"])

    print("Tree:")

    tree = g.nodes[explanation]["tree"]
print_explanation(explanation)

visualize_local_generalization_graph(g, explanation)
