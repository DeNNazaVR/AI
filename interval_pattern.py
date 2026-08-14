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

class Pattern:
    segments = []
    def __init__(self, segments1):
        self.segments = segments1
    def __str__(self):
        s = ", ".join(str(s) for s in self.segments)
        return f"Pattern(segments=[{s}])"
    
    def __repr__(self):
        return f"Pattern({self.segments})"




v1 = Segment(3, 4)
v2 = Segment(3, 5)
a = [v1, v2]
p = Pattern(a)
print(p)