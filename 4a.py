# Importing required library
import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('ch1ex1.csv')
points = df.values  # Convert dataframe to NumPy array

# Step 2: Import KMeans and create the model
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)  # Initialize KMeans with 3 clusters
model.fit(points)             # Fit the model to the data
labels = model.predict(points)  # Predict cluster labels for each data point

# Step 3: Import matplotlib for visualization
import matplotlib.pyplot as plt

# Extract X and Y coordinates from the dataset
xs = points[:, 0]
ys = points[:, 1]

# Step 4: Plot clustered data points
plt.scatter(xs, ys, c=labels)
plt.title("K-Means Clustering - Data Points")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 5: Get cluster centroids
centroids = model.cluster_centers_

# Extract centroid coordinates
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

# Step 6: Plot clusters with centroids
plt.scatter(xs, ys, c=labels)
plt.scatter(centroids_x, centroids_y, marker='X', s=200, c='black')  # Centroids as 'X'
plt.title("K-Means Clustering with Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
