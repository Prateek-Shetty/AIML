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


# Step 4: Import fcluster


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