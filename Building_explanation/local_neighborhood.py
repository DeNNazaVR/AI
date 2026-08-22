import numpy as np

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
    

class Object:
    features = []
    def __init__(self, features1):
        self.features = features1
    def __repr__(self):
        return f"Object({self.features})"


def point_to_pattern(p):
    segs = []
    for el in p:
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


def gen_neighboors(x, feature_std, n_samples=1000,sigma=0.1,seed=14131232562):
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

objc = [2, 3, 4]
o = Object(objc)
feature_std = np.array([1, 2.4, 0.7])

vals = gen_neighboors(o, feature_std)
print(vals)
# s1 = Segment(1, 6)
# s2 = Segment(-1, 24)
# s3 = Segment(4, 4)
# p1= Pattern([s1, s2, s3])
# s1 = Segment(1, 6)
# s2 = Segment(-1, 4)
# s3 = Segment(4, 4)
# p2 = Pattern([s1, s2, s3])
# print(is_more_general(p1, p2))