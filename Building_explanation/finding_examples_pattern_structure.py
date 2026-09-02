from explanation_pattern_structure import Object, model, x, x_train, x_test


INF32 = 1e9
mx = -INF32
mn = INF32
ind1 = 0
ind2 = 0
ind = 0
for i in x_test:
    proba = model.predict_proba([i])[0]
    d = abs(proba[0] - proba[1])
    if d > mx:
        mx = d
        ind1 = ind
    if d < mn:
        mn = d
        ind2 = ind
    ind += 1
print(mx)
print(ind1)
print(mn)
print(ind2)
mn = INF32
need = 0.64
ind3 = 0
ind = 0
for i in x_test:
    proba = model.predict_proba([i])[0]
    d = abs(proba[0] - proba[1])
    d1 = abs(d - need)
    if d1 < mn:
        mn = d1
        ind3 = ind
    ind += 1
print(mn)
print(ind3)