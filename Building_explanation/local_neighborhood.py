import numpy as np
from interval_pattern import *




def gen_neighboors(x, feature_std, n_samples=1000,sigma=1,seed=1312562):
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