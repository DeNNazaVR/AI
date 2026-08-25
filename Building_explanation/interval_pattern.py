
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


def object_to_pattern(o):
    segs = []
    for el in o.features:
        s = Segment(el, el)
        segs.append(s)
    ans = Pattern(segs)
    return ans

def meet(p1, p2):
    segs = []
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
    ans = Pattern(segs)
    return ans

def contains_object(p, o):
    if len(p.segments) < len(o.features):
        return False
    for i in range(len(o.features)):
        if not p.segments[i].is_point_in(o.features[i]):
            return False
    return True

def extent(p, local_samples):
    ans = []
    for obj in local_samples:
        if contains_object(p, obj):
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




# objc = [2, 3, 4]
# s1 = Segment(1, 6)
# s2 = Segment(-1, 24)
# s3 = Segment(4, 4)
# p1= Pattern([s1, s2, s3])
# s1 = Segment(1, 6)
# s2 = Segment(-1, 4)
# s3 = Segment(4, 4)
# p2 = Pattern([s1, s2, s3])
# print(is_more_general(p1, p2))