import numpy as np
import random
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

random_seed = 1312562
random.seed(random_seed)

#black box model trainings


x, y = make_classification(
    n_samples=1000,
    n_classes=2,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    class_sep=0.15,
    flip_y=0.1,
    random_state=random_seed
)


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


class Segment:
    l = 0
    r = 0
    def __init__(self, l1 = 0, r1 = 0):
        self.l = l1
        self.r = r1

    def __str__(self):
        return f"Segment(l={self.l}, r={self.r})"
    def __repr__(self):
        return f"Segment({self.l}, {self.r})"
    def is_point_in(self, p):
        return self.l <= p <= self.r
    def is_segment_in(self, s):
        return self.l <= s.l <= s.r <= self.r


class Pattern:
    segments = []
    def __init__(self, segments1):
        self.segments = segments1
    def __str__(self):
        s = ", ".join(str(s) for s in self.segments)
        return f"Pattern(segments=[{s}])"
    
    def __repr__(self):
        return f"Pattern({self.segments})"
    
    def __eq__(self, other):
        if not isinstance(other, Pattern):
            return False

        if len(self.segments) != len(other.segments):
            return False

        for i in range(len(self.segments)):
            if self.segments[i].l != other.segments[i].l:
                return False
            if self.segments[i].r != other.segments[i].r:
                return False

        return True
    
    def __hash__(self):
        vals = []

        for s in self.segments:
            vals.append((s.l, s.r))

        return hash(tuple(vals))

class Object:
    features = []
    def __init__(self, features1):
        self.features = features1
    def __repr__(self):
        return f"Object({self.features})"


def object_to_pattern(o):
    segs = []
    for el in o.features:
        s = Segment(el, el)
        segs.append(s)
    ans = Pattern(segs)
    return ans

def meet(p1, p2):
    segs = []
    if len(p1.segments) > len(p2.segments):
        p1, p2 = p2, p1
    sz = len(p1.segments)
    for i in range(sz):
        l1 = p1.segments[i].l
        r1 = p1.segments[i].r
        l2 = p2.segments[i].l
        r2 = p2.segments[i].r
        l = min(l1, l2)
        r = max(r1, r2)
        s = Segment(l, r)
        segs.append(s)
    for i in range(sz, len(p2.segments)):
        segs.append(p2.segments[i])
    ans = Pattern(segs)
    return ans

def covers(p, o):
    if len(p.segments) < len(o.features):
        return False
    for i in range(len(o.features)):
        if not p.segments[i].is_point_in(o.features[i]):
            return False
    return True

def extent(p, local_samples):
    ans = []
    for obj in local_samples:
        if covers(p, obj):
            ans.append(obj)
    return ans

def closure(p, local_samples):
    vals = extent(p, local_samples)
    segs = []
    mx = 0
    for obj in vals:
        mx = max(mx, len(obj.features))
    for i in range(mx):
        was = 0
        l = 0
        r = 0
        for obj in vals:
            if len(obj.features) > i:
                if was == 0:
                   l = obj.features[i]
                   r = obj.features[i]
                   was = 1
                else:
                   l = min(l, obj.features[i])
                   r = max(r, obj.features[i])
        s = Segment(l, r)
        segs.append(s)
    return Pattern(segs)

def is_more_general(p1, p2):
    if len(p1.segments) < len(p2.segments):
        return False
    sz = len(p2.segments)
    for i in range(sz):
        if not p1.segments[i].is_segment_in(p2.segments[i]):
            return False
    return True


def gen_neighboors(x, feature_std, n_samples=1000,sigma=1,seed=random_seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(
        loc=0,
        scale=sigma * feature_std,
        size=(n_samples, len(x.features))
    )
    x1 = np.array(x.features)
    samples = x1 + noise
    ans = []
    for i in samples:
        o = Object(i.tolist())
        ans.append(o)
    return ans


def get_dist(x, local_samples, feature_std):
    dist = []
    x1 = np.array(x.features)
    for o in local_samples:
        z = np.array(o.features)
        dif = (z - x1) / feature_std
        distance = np.sqrt(
            np.sum(dif ** 2)
        )
        dist.append(distance)

    return np.array(dist)


def get_weights(dist, sigma):
    ans = np.exp(
        -(dist ** 2) / (sigma ** 2)
    )
    return ans

def calc_purity(p, local_samples, local_predictions, weights, o):
    ind = 0
    s = 0
    our_f = f(o)
    have = 0
    for z in local_samples:
        if not covers(p, z):
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
    cnt = 20
    ans = []
    n = len(local_samples)
    for i in range(cnt):
        take = []
        k = random.randint(1, 20)
        for j in range(k):
            ind = random.randint(1, 30)
            while ind in take:
                ind = random.randint(1, n)
            take.append(ind)
        p = object_to_pattern(o)
        for ind in take:
            p = meet(p, object_to_pattern(local_samples[ind - 1]))
        p = closure(p, local_samples)
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
need_purity = 0.5
need_support = 10

def get_final_explanation(g, need_purity, need_support):
    take = []
    for node in g.nodes:
        purity = g.nodes[node]["purity"]
        support = g.nodes[node]["support"]
        take.append(node)
    mx_depth_nodes = []
    for node in take:
        can_generalize = False
        for next_node in g.successors(node):
            if next_node in take:
                can_generalize = True
                break
        if not can_generalize:
            mx_depth_nodes.append(node)
    if len(mx_depth_nodes) == 0:
        return None
    best = mx_depth_nodes[0]

    for node in mx_depth_nodes:
        if g.nodes[node]["support"] > g.nodes[best]["support"]:
            best = node
        elif g.nodes[node]["support"] == g.nodes[best]["support"]:
            if g.nodes[node]["purity"] > g.nodes[best]["purity"]:
                best = node

    return best


feature_std = np.std(x_train, axis=0)

features_len = 5
features =  [4, -1, 8, -0.57, 1]
o = Object(features)

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