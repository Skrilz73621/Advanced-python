import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. Загрузка датасета Kaggle
data = pd.read_csv("data.csv")

# 2. Убираем ненужные колонки
data = data.drop(['id', 'Unnamed: 32'], axis=1)

# 3. Разделение признаков и целевой переменной
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# 4. Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 6. t-SNE с уменьшенным количеством итераций
tsne = TSNE(n_components=2, perplexity=20, n_iter=500, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 7. Визуализация
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="Set1", legend="full")
plt.title("PCA: Breast Cancer Dataset")

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="Set1", legend="full")
plt.title("t-SNE: Breast Cancer Dataset")

plt.tight_layout()
plt.show()

# 8. Пояснение
print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)
