from explanation import Object, model, x_train


INF32 = 1e9
# mx = -INF32
# mn = INF32
# o1 = Object([])
# o2 = Object([])
# for i in x_train:
#     proba = model.predict_proba([i])[0]
#     d = abs(proba[0] - proba[1])
#     if d > mx:
#         mx = d
#         o1 = Object(i.tolist())
#     if d < mn:
#         mn = d
#         o2 = Object(i.tolist())
# print(o1.features)
# print(mx)
# print(o2)
# print(mn) 
mn = INF32
o1 = Object([])
for i in x_train:
    proba = model.predict_proba([i])[0]
    d = abs(proba[0] - proba[1])
    d1 = abs(0.45 - d)
    if d1 < mn:
        mn = d1
        o1 = Object(i.tolist())
print(o1)
print(mn)