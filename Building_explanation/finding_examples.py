from explanation import Object, model, x_train


INF32 = 1e9
mx = -INF32
mn = INF32
ind1 = 0
ind2 = 0
ind = 0
for i in x_train:
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