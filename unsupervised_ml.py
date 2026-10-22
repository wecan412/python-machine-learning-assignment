import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load the dataset
# NOTE: Replace 'Mall_Customers.csv' with your actual file name.
try:
    # Update column names based on your specific dataset
    df = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    print("Error: Dataset not found. Please ensure the CSV is in the same directory.")
    exit()

# 2. Prepare the data
# For this example, let's take two numerical columns for 2D plotting
# You will need to change these indices based on your dataset
X = df.iloc[:, [3, 4]].values 

# 3. Use the Elbow Method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# 4. Train K-Means with the chosen number of clusters (e.g., 5)
# You should look at the Elbow graph to decide the number. We'll use 5 as an example.
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# 5. Visualize the clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids', marker='X')
plt.title('K-Means Clustering of Customers')
plt.xlabel('Feature 1 (e.g., Annual Income)')
plt.ylabel('Feature 2 (e.g., Spending Score)')
plt.legend()
plt.grid(True)
plt.show()
