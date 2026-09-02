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
from sklearn.tree import DecisionTreeClassifier

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
    def __init__(self, ind=-1, gr=0, left=False):
        self.ind = ind
        self.gr = gr
        self.left = left

    def __repr__(self):
        return (
            f"Node("
            f"feature={self.ind}, "
            f"gr={self.gr}, "
            f"left={self.left}"
            f")"
        )

def get_path(tree, o):
    ans = []

    path = tree.decision_path([o.features])

    for node_ind in path.indices:
        left = tree.tree_.children_left[node_ind]
        right = tree.tree_.children_right[node_ind]

        if left == right:
            continue

        ind = tree.tree_.feature[node_ind]
        gr = tree.tree_.threshold[node_ind]

        left = o.features[ind] <= gr

        node = Node(
            ind=ind,
            gr=gr,
            left=left
        )

        ans.append(node)

    return ans

def contains_object(tree, o, z):
    path = get_path(tree, o)
    for node in path:
        if (o.features[node.ind] <= node.gr) != (z.features[node.ind] <= node.gr):
            return False
    return True 

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

feature_std = np.std(x_train, axis=0)

test_ind = 43
o = Object(x_test[test_ind].tolist())

local_samples = gen_neighboors(o, feature_std)
dist = get_dist(o, local_samples, feature_std)
order = np.argsort(dist)
local_samples = [
    local_samples[i]
    for i in order
]
dist = dist[order]

local_features = []
for obj in local_samples:
    local_features.append(obj.features)

local_features = np.array(local_features)
local_predictions = model.predict(local_features)
weights = get_weights(dist, 2)


tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=random_seed
)

tree.fit(
    local_features,
    local_predictions,
    sample_weight=weights
)

def print_explanation(tree, o, local_samples, local_predictions, weights):
    path = get_path(tree, o)

    print()
    print("========== DECISION TREE EXPLANATION ==========")
    print()

    print("Black-box prediction:", target_names[f(o)])
    print("Tree prediction:", target_names[tree.predict([o.features])[0]])

    print()
    print("Explanation:")
    print()

    for i in range(len(path)):
        node = path[i]

        if i != 0:
            print("AND")

        if node.left:
            print(
                f"{feature_names[node.ind]} <= {node.gr:.6f}"
            )
        else:
            print(
                f"{feature_names[node.ind]} > {node.gr:.6f}"
            )

    support = len(extent(tree, o, local_samples))
    coverage = support / len(local_samples)

    purity = calc_purity(
        tree,
        local_samples,
        local_predictions,
        weights,
        o
    )

    print()
    print("-----------------------------------------------")
    print(f"Support:  {support}")
    print(f"Coverage: {coverage:.4f}")
    print(f"Purity:   {purity:.4f}")
    print("===============================================")

print_explanation(tree, o, local_samples, local_predictions, weights)

