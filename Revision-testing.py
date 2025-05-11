from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
x,y =  make_classification(n_samples= 200, n_features=10)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(x,y)

clusters = KMeans(n_clusters=2).fit_predict(x)

pca = PCA(n_components=2)

X_reduced = pca.fit_transform(x)


import matplotlib.pyplot as plt

# True labels
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='bwr', edgecolor='k')
plt.title('True Labels')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

# KMeans clusters
plt.subplot(1, 3, 2)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='bwr', edgecolor='k')
plt.title('KMeans Clusters')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

# Logistic Regression predictions
preds = model.predict(x)
plt.subplot(1, 3, 3)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=preds, cmap='bwr', edgecolor='k')
plt.title('Logistic Regression Predictions')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

plt.tight_layout()
plt.show()