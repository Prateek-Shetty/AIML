"""Hierarchical Clustering Algorithm on seeds_less_rows dataset for extracting cluster  labels of different varieties of seeds """


import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

seeds_df = pd.read_csv('seeds-less-rows.csv')

varieties = list(seeds_df.pop('grain_variety'))

samples = seeds_df.values

mergings = linkage(samples, method='complete')

plt.figure(figsize=(10, 6))
dendrogram(mergings, labels=varieties, leaf_rotation=90, leaf_font_size=6)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index or Variety")
plt.ylabel("Distance")
plt.show()



# Step 5: Extract flat clusters with a maximum height of 6
labels = fcluster(mergings, 6, criterion='distance')

# Step 6: Create a DataFrame with cluster labels and original varieties
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Step 7: Create a cross-tabulation between cluster labels and grain varieties
ct = pd.crosstab(df['labels'], df['varieties'])

# Step 8: Display the cross-tabulation
print("\nCross-tabulation of Cluster Labels vs Grain Varieties:")
print(ct)


"""Conclusion: Three varieties of labels extracted from 'seeds-less-rows’ dataset by applying  Hierarchical clustering technique as shown in the output table."""





# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Step 2: Create or Load Dataset
# Example data (you can replace this with your own CSV)
data = {
    'X': [1, 2, 3, 5, 6, 7, 8, 25, 26, 27],
    'Y': [1, 1, 2, 5, 6, 7, 7, 25, 26, 27]
}
df = pd.DataFrame(data)

# Step 3: Visualize Data
plt.scatter(df['X'], df['Y'])
plt.title('Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Step 4: Create Dendrogram
plt.figure(figsize=(8, 5))
dendrogram = sch.dendrogram(sch.linkage(df, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

# Step 5: Fit Hierarchical Clustering Model
hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels = hc.fit_predict(df)

# Step 6: Visualize Clusters
plt.figure(figsize=(6, 5))
plt.scatter(df['X'], df['Y'], c=labels, cmap='rainbow')
plt.title('Clusters Formed')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Step 7: Display Results
df['Cluster'] = labels
print(df)
