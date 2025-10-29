# Step 1 & 2: Importing libraries and loading the dataset
import pandas as pd

# Load dataset
seeds_df = pd.read_csv('seeds-less-rows.csv')

# Remove the grain species column for clustering (save it for comparison)
varieties = list(seeds_df.pop('grain_variety'))

# Extract the numerical measurements as a NumPy array
samples = seeds_df.values

# Step 3: Run hierarchical clustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Perform hierarchical clustering using the 'complete' linkage method
mergings = linkage(samples, method='complete')

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(mergings, labels=varieties, leaf_rotation=90, leaf_font_size=6)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index or Variety")
plt.ylabel("Distance")
plt.show()

# Step 4: Import fcluster
from scipy.cluster.hierarchy import fcluster

# Step 5: Extract flat clusters with a maximum height of 6
labels = fcluster(mergings, 6, criterion='distance')

# Step 6: Create a DataFrame with cluster labels and original varieties
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Step 7: Create a cross-tabulation between cluster labels and grain varieties
ct = pd.crosstab(df['labels'], df['varieties'])

# Step 8: Display the cross-tabulation
print("\nCross-tabulation of Cluster Labels vs Grain Varieties:")
print(ct)
