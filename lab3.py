# -------------------------------------------------------------
# Import Libraries
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial import distance

# -------------------------------------------------------------
# Load Dataset (Iris, but only take 2 classes: Setosa & Versicolor)
# -------------------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target

# Keep only first two classes (binary classification)
mask = y < 2
X = X[mask]
y = y[mask]

# -------------------------------------------------------------
# A1. Evaluate intraclass spread and interclass distance
# -------------------------------------------------------------
X_class0 = X[y == 0]
X_class1 = X[y == 1]

# Centroids (means)
centroid0 = X_class0.mean(axis=0)
centroid1 = X_class1.mean(axis=0)

# Spread (standard deviation)
spread0 = X_class0.std(axis=0)
spread1 = X_class1.std(axis=0)

# Distance between centroids
dist_centroids = np.linalg.norm(centroid0 - centroid1)

print("\n#A1 Results")
print("Centroid Class 0:", centroid0)
print("Centroid Class 1:", centroid1)
print("Spread Class 0:", spread0)
print("Spread Class 1:", spread1)
print("Distance between centroids:", dist_centroids)

# -------------------------------------------------------------
# A2. Take one feature, plot histogram, calculate mean & variance
# -------------------------------------------------------------
feature_index = 0  # sepal length (first column)
feature_data = X[:, feature_index]

plt.hist(feature_data, bins=10, color="skyblue", edgecolor="black")
plt.title("Histogram of Feature: Sepal Length")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

mean_val = np.mean(feature_data)
var_val = np.var(feature_data)

print("\n#A2 Results")
print("Mean of feature:", mean_val)
print("Variance of feature:", var_val)

# -------------------------------------------------------------
# A3. Minkowski distance between two feature vectors (r=1 to 10)
# -------------------------------------------------------------
vec1 = X[0]
vec2 = X[1]

minkowski_distances = []
r_values = list(range(1, 11))

for r in r_values:
    dist = distance.minkowski(vec1, vec2, r)
    minkowski_distances.append(dist)

plt.plot(r_values, minkowski_distances, marker='o')
plt.title("Minkowski Distance (r=1 to 10)")
plt.xlabel("r value")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

print("\n#A3 Results")
print("Minkowski Distances (r=1..10):", minkowski_distances)

# -------------------------------------------------------------
# A4. Divide dataset into train and test sets
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\n#A4 Results")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------------------------------------------------------------
# A5. Train kNN classifier with k=3
# -------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print("\n#A5 Results")
print("kNN model trained with k=3")

# -------------------------------------------------------------
# A6. Test accuracy of kNN on test set
# -------------------------------------------------------------
accuracy = knn.score(X_test, y_test)
print("\n#A6 Results")
print("kNN Accuracy on test set (k=3):", accuracy)

# -------------------------------------------------------------
# A7. Prediction behavior of kNN
# -------------------------------------------------------------
predictions = knn.predict(X_test)
print("\n#A7 Results")
print("Predictions on test set:", predictions)

# Predict for a single sample
sample = X_test[0].reshape(1, -1)
pred = knn.predict(sample)
print("Prediction for first test vector:", pred)

# -------------------------------------------------------------
# A8. Vary k from 1 to 11 and plot accuracy
# -------------------------------------------------------------
accuracies = []
k_values = list(range(1, 12))

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    accuracies.append(acc)

plt.plot(k_values, accuracies, marker='o')
plt.title("Accuracy vs k in kNN")
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

print("\n#A8 Results")
for k, acc in zip(k_values, accuracies):
    print(f"k={k}: Accuracy={acc:.3f}")

# -------------------------------------------------------------
# A9. Confusion Matrix & Classification Report
# -------------------------------------------------------------
y_pred = knn.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names[:2])

print("\n#A9 Results")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
