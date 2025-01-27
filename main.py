########################################################
# E2E Implementation of EDA, Lookalike Model & Clustering
# Amit Ram Shinde  CSE IIT GOA  
########################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports for modeling & metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

# ------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------

# Adjust the file paths to match your local file structure
customers = pd.read_csv('Customers.csv')      # e.g. columns: CustomerID, CustomerName, Region, SignupDate
products = pd.read_csv('Products.csv')        # e.g. columns: ProductID, ProductName, Category, Price
transactions = pd.read_csv('Transactions.csv') # e.g. columns: TransactionID, CustomerID, ProductID, TransactionDate,
                                              #               Quantity, TotalValue, Price

# Quick checks
print("Customers head:")
print(customers.head(), "\n")
print("Products head:")
print(products.head(), "\n")
print("Transactions head:")
print(transactions.head(), "\n")

# Convert date columns to datetime if needed
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'], errors='coerce')
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'], errors='coerce')

# ------------------------------------------------------
# 2. EDA & BUSINESS INSIGHTS (Task 1)
# ------------------------------------------------------

# 2.1 Merge data into a single DataFrame for EDA
df_merged = transactions.merge(customers, on='CustomerID', how='left') \
                        .merge(products, on='ProductID', how='left')

# Basic summary
print("\nMerged DF Info:")
print(df_merged.info())
print(df_merged.describe(include='all'))

# Check for missing values
print("\nMissing values in merged DF:")
print(df_merged.isnull().sum())

# Example EDA: distribution of TotalValue
plt.figure(figsize=(6,4))
sns.histplot(df_merged['TotalValue'], bins=30, kde=True)
plt.title("Distribution of Total Transaction Values")
plt.show()

# Example EDA: top categories by total revenue
rev_by_cat = df_merged.groupby('Category')['TotalValue'].sum().reset_index()
rev_by_cat = rev_by_cat.sort_values('TotalValue', ascending=False)
print("\nTop categories by total revenue:\n", rev_by_cat)

# Example EDA: total revenue by region
rev_by_region = df_merged.groupby('Region')['TotalValue'].sum().reset_index()
rev_by_region = rev_by_region.sort_values('TotalValue', ascending=False)
print("\nTotal revenue by region:\n", rev_by_region)

# More plots or stats as needed...
# ...

# -- INSIGHTS (You’d normally put these into a PDF) --
# For example:
# 1) "Category X accounts for 40% of total revenue."
# 2) "Region Y contributes the highest average order value."
# 3) etc...

# ------------------------------------------------------
# 3. LOOKALIKE MODEL (Task 2)
# ------------------------------------------------------
# We want to recommend top-3 similar customers to each user,
# using both customer profile + transaction history.

## 3.1 Create Customer-Level Features

# 3.1.1 Transaction aggregations per customer
agg_trans = df_merged.groupby('CustomerID').agg(
    total_spend = ('TotalValue','sum'),
    avg_spend   = ('TotalValue','mean'),
    total_orders= ('TransactionID','count')
).reset_index()

# 3.1.2 Favorite category (or category distribution)
# Let's do a simple approach: the top category for each customer
fav_cat_df = df_merged.groupby(['CustomerID','Category'])['TotalValue'].sum().reset_index()
fav_cat_df['rank'] = fav_cat_df.groupby('CustomerID')['TotalValue'].rank(method='first', ascending=False)
fav_cat_df = fav_cat_df[fav_cat_df['rank']==1].drop(columns='rank')
fav_cat_df.rename(columns={'Category':'FavoriteCategory'}, inplace=True)

# 3.1.3 Merge these features back with Customers
cust_features = customers.merge(agg_trans, on='CustomerID', how='left') \
                         .merge(fav_cat_df[['CustomerID','FavoriteCategory']], on='CustomerID', how='left')

# Fill any missing numeric values (for new customers etc.)
cust_features[['total_spend','avg_spend','total_orders']] = cust_features[['total_spend','avg_spend','total_orders']].fillna(0)

# 3.1.4 Encode categorical fields (Region, FavoriteCategory)

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_cols = ['Region','FavoriteCategory']
encoded_cat = ohe.fit_transform(cust_features[cat_cols].fillna('Unknown'))


cat_feature_names = ohe.get_feature_names_out(cat_cols)
encoded_cat_df = pd.DataFrame(encoded_cat, columns=cat_feature_names)

# Combine numeric + encoded categorical
numeric_cols = ['total_spend','avg_spend','total_orders']
final_df = pd.concat([cust_features[['CustomerID'] + numeric_cols], encoded_cat_df], axis=1)

# 3.1.5 Scaling
scaler = StandardScaler()
scaled_vals = scaler.fit_transform(final_df[numeric_cols + list(cat_feature_names)])
# Put scaled data back in a DataFrame
scaled_df = pd.DataFrame(scaled_vals, columns=numeric_cols + list(cat_feature_names))

# We'll keep track of the CustomerID separately
scaled_df['CustomerID'] = final_df['CustomerID']

# 3.2 Similarity Calculation
# Create a matrix in same order as scaled_df
feature_matrix = scaled_df.drop(columns='CustomerID').values  # shape (N_customers, n_features)
similarity_mat = cosine_similarity(feature_matrix)

# 3.3 For each of the first 20 customers (C0001–C0020), find top-3 lookalikes
all_customer_ids = scaled_df['CustomerID'].tolist()

# Utility to get index by ID:
cust_id_to_index = {cid: idx for idx, cid in enumerate(all_customer_ids)}

lookalike_results = []

for cid in all_customer_ids:
    # We only do for first 20 if cid in [C0001..C0020], but let's illustrate for all
    # parse the "Cxxxx" numeric part if needed, or just filter some other way.
    # For demonstration, we'll do all:

    cindex = cust_id_to_index[cid]
    sim_scores = similarity_mat[cindex]  # 1D array of similarity to every customer
    # create pairs (other_cid, score)
    pairs = [(all_customer_ids[i], sim_scores[i]) for i in range(len(all_customer_ids)) if i != cindex]
    # sort by similarity descending
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    top3 = pairs_sorted[:3]
    # store results
    lookalike_results.append({
        'SourceCustomerID': cid,
        'RecommendedCust1': top3[0][0],
        'Score1': top3[0][1],
        'RecommendedCust2': top3[1][0],
        'Score2': top3[1][1],
        'RecommendedCust3': top3[2][0],
        'Score3': top3[2][1]
    })

lookalike_df = pd.DataFrame(lookalike_results)

# You could filter only those rows for C0001..C0020 if you wish, e.g.:
# lookalike_df = lookalike_df[lookalike_df['SourceCustomerID'].isin(['C0001','C0002', ... 'C0020'])]

# 3.4 Save lookalikes to CSV
lookalike_df.to_csv('Lookalike.csv', index=False)
print("\nSample from Lookalike.csv results:\n", lookalike_df.head())

# ------------------------------------------------------
# 4. CUSTOMER SEGMENTATION / CLUSTERING (Task 3)
# ------------------------------------------------------

# We'll reuse the 'scaled_df' as the input for clustering
# Because it has the numeric/categorical features combined and scaled.

X = scaled_df.drop(columns='CustomerID').values

# 4.1 We can try KMeans for multiple k values to find best DB index
possible_ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
db_scores = {}

for k in possible_ks:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    db_index = davies_bouldin_score(X, labels)
    db_scores[k] = db_index
    print(f"k={k}, DB Index={db_index:.3f}")

# Pick a k with the lowest DB Index or your chosen metric
best_k = min(db_scores, key=db_scores.get)
print(f"\nChosen number of clusters (best_k): {best_k} with DB Index={db_scores[best_k]:.3f}")

# 4.2 Fit final KMeans with best_k
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
labels_final = kmeans_final.fit_predict(X)

# Attach cluster labels to the scaled_df
scaled_df['ClusterLabel'] = labels_final

# 4.3 Example: Visualize Clusters in 2D using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)  # shape (N_customers, 2)

plt.figure(figsize=(6,5))
plt.scatter(coords[:,0], coords[:,1], c=labels_final, cmap='viridis', alpha=0.7)
plt.title(f"KMeans Clusters (k={best_k}) - PCA 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.show()

# 4.4 Summarize your clusters
cluster_summary = scaled_df.groupby('ClusterLabel').mean(numeric_only=True)

print("\nCluster Summary (mean of scaled features):")
print(cluster_summary)

# In your final PDF, you might describe each cluster,
# e.g. "Cluster 0: High spenders mostly from Region X..."

# ------------------------------------------------------
# END OF SCRIPT
# ------------------------------------------------------
