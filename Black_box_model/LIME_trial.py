from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import random


x, y = make_classification(
    n_samples=1000,
    n_classes=2,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    class_sep=0.15,
    flip_y=0.1,
    random_state=4254
)


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.5,
    random_state=61176,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=4452
)

model.fit(x_train, y_train)

train_vals = model.predict(x_train)
test_vals = model.predict(x_test)

print("Train accuracy:", accuracy_score(y_train, train_vals))
print("Test accuracy:", accuracy_score(y_test, test_vals))
print("Confusion matrix:")
print(confusion_matrix(y_test, test_vals))



names = [
    "x1", "x2", "x3", "x4", "x5",
    "x6", "x7", "x8", "x9", "x10"
]

expl = LimeTabularExplainer(
    x_train,
    feature_names=names,
    class_names=["class 0", "class 1"],
    mode="classification",
    random_state=42
)

print()
random.seed(4234701732569832)
take_cnt = 10
take_inds = []
for i in range(take_cnt):
    cur = random.randint(0, len(y_test))
    while cur in take_inds:
        cur = random.randint(0, len(y_test))
    take_inds.append(cur)

for i in take_inds:
    print("testing for i = i", i)
    v1 = model.predict([x_test[i]])[0]
    v2 = y_test[i]
    print("Right class:", v2)
    print("Your class:", v1)
    probs = model.predict_proba([x_test[i]])[0]
    d = abs(probs[0] - probs[1])
    if v1 == v2:
        print("OK")
        print(d)
    else:
        print("WA")
        print(d)
    print()
    print("Probs:", model.predict_proba([x_test[i]])[0])

    ans = expl.explain_instance(
        x_test[i],
        model.predict_proba,
        num_features=10
    )

    for param, impact in ans.as_list():
        print(param, impact)
    print()