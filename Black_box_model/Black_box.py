from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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
    test_size=0.2,
    random_state=42,
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