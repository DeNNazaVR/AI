import numpy as np
import random
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from interval_pattern import *
from local_neighborhood import *

random_seed = 1312562
random.seed(random_seed)

#black box model trainings


data = load_breast_cancer()

x = data.data
y = data.target

feature_names = data.feature_names
target_names = data.target_names
# mns = [1e9] * 5
# mxs = [-1e9] * 5
# for i in x:
#     for j in range(5):
#         mns[j] = min(mns[j], i[j])
#         mxs[j] = max(mxs[j], i[j])
# print(mns)
# print(mxs)

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


#function f, we want to explain
def f(obj):
    vals = np.array([obj.features])
    ans = model.predict(vals)
    return ans[0]


def calc_purity(p, local_samples, local_predictions, weights, o):
    ind = 0
    s = 0
    our_f = f(o)
    have = 0
    for z in local_samples:
        if not contains_object(p, z):
            ind += 1
            continue
        s += weights[ind]
        if local_predictions[ind] == our_f:
            have += weights[ind]
        ind += 1
    if s == 0:
        return 0
    return have / s

def build_candidate_patterns(o, local_samples):
    cnt = 60
    ans = []
    n = len(local_samples)
    k_range = 50
    ind_range = 200
    for i in range(cnt):
        take = []
        k = random.randint(1, k_range)
        for j in range(k):
            ind = random.randint(1, ind_range)
            while ind in take:
                ind = random.randint(1, ind_range)
            take.append(ind)
        p = object_to_pattern(o)
        for ind in take:
            p = meet(p, object_to_pattern(local_samples[ind - 1]))
        p = closure(p, local_samples + [o])
        ans.append(p)
    #trying to add more randomised patterns
    cnt = 30
    k_range = 100
    ind_range = 500
    for i in range(cnt):
        take = []
        k = random.randint(1, k_range)
        for j in range(k):
            ind = random.randint(1, ind_range)
            while ind in take:
                ind = random.randint(1, ind_range)
            take.append(ind)
        p = object_to_pattern(o)
        for ind in take:
            p = meet(p, object_to_pattern(local_samples[ind - 1]))
        p = closure(p, local_samples + [o])
        ans.append(p)
    K = 8
    for mask in range(1 << K):
        p = object_to_pattern(o)
        for i in range(K):
            if (mask >> i) & 1:
                p = meet(p, object_to_pattern(local_samples[i]))
        p = closure(p, local_samples + [o])
        ans.append(p)
    lim_len = 12
    for I in range(1,lim_len + 1):
        cnt = min(10, (n + i - 1) // i)

        for i in range(cnt):
            take = []
            k = I
            for j in range(k):
                ind = random.randint(1, ind_range)
                while ind in take:
                    ind = random.randint(1, ind_range)
                take.append(ind)
            p = object_to_pattern(o)
            for ind in take:
                p = meet(p, object_to_pattern(local_samples[ind - 1]))
            p = closure(p, local_samples + [o])
            ans.append(p)
    return ans



def get_useful_candidate_patterns(candidate_patterns):
    dif = set()
    for p in candidate_patterns:
        dif.add(p)
    candidate_patterns = []
    for el in dif:
        candidate_patterns.append(el)


    return candidate_patterns


def build_local_generalization_graph(candidate_patterns, local_samples, local_predictions, weights, o):
    g = nx.DiGraph()
    N = len(candidate_patterns)
    for i in range(N):
        p = candidate_patterns[i]
        a = extent(p, local_samples)
        support = len(a)
        purity = calc_purity(p, local_samples, local_predictions, weights, o)
        coverage = support / len(local_samples)

        description = []
        for j in range(len(p.segments)):
            description.append((p.segments[j].l, p.segments[j].r))

        g.add_node(
            i,
            pattern=p,
            support=support,
            coverage=coverage,
            purity=purity,
            predicted_class=f(o),
            description=description
        )

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            p_spec = candidate_patterns[i]
            p_gen = candidate_patterns[j]
            if not is_more_general(p_gen, p_spec):
                continue
            if is_more_general(p_spec, p_gen):
                continue
            direct = True
            for k in range(N):
                if k == i or k == j:
                    continue
                p_check = candidate_patterns[k]
                v1 = is_more_general(p_check, p_spec) and not is_more_general(p_spec, p_check)
                v2 = is_more_general(p_gen, p_check) and not is_more_general(p_check, p_gen)
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

need_purity = 0.6
need_support = 80

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


s1 = 0
s2 = 0
for i in weights:
    s1 += i
    s2 += i * i
s1 *= s1
ess = s1 / s2
print("ess is: ", ess)

candidate_patterns = build_candidate_patterns(o, local_samples)
candidate_patterns = get_useful_candidate_patterns(candidate_patterns)



g = build_local_generalization_graph(candidate_patterns, local_samples, local_predictions, weights, o)
explanation = get_final_explanation(g, need_purity, need_support)
def print_explanation(explanation):
    print("Node:", explanation)
    print("Pattern:", g.nodes[explanation]["pattern"])
    print("Support:", g.nodes[explanation]["support"])
    print("Coverage:", g.nodes[explanation]["coverage"])
    print("Purity:", g.nodes[explanation]["purity"])
    print("Predicted class:", g.nodes[explanation]["predicted_class"])
    print("Description:", g.nodes[explanation]["description"])
print_explanation(explanation)

visualize_local_generalization_graph(g, explanation)

