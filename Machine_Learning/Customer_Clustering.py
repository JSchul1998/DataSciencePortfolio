import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
## Display all the columns of the dataframe
pd.pandas.set_option('display.max_columns',None)
from scipy import stats
from scipy.stats import norm, skew 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
# clustering algorithms
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_samples, silhouette_score

customer_df = pd.read_csv("Desktop/Mall_Customers.csv")

print(customer_df.head(10))
print(customer_df.describe())
customer_df.Gender = customer_df.Gender.astype('category')

customer_dtype = customer_df.dtypes
print(customer_dtype.value_counts())

print(customer_df.info())

sns.pairplot(customer_df, vars=["Age", "Annual Income (k$)", "Spending Score (1-100)"],  kind ="reg", hue = "Gender", palette="husl", markers = ['o','D'])
plt.show()

customer_df.drop(columns='CustomerID',axis=1,inplace=True)
customer_df = pd.get_dummies(customer_df).reset_index(drop=True)
print(customer_df.head(10))


### Model Development

## Tuning Hyperparameter (k) using the Elbow Method
inert = []
range_val = range(1,15)
for i in range_val:
  kmean = KMeans(n_clusters=i)
  kmean.fit_predict(customer_df)
  inert.append(kmean.inertia_)
plt.plot(range_val,inert,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show()
## Appears as though 5 is an optimal number of clusters 


# Kmeans algorithm
kmeans_model=KMeans(n_clusters=5)
kmeans_clusters = kmeans_model.fit_predict(customer_df)
# Agglomerative algorithm
agglo_model = AgglomerativeClustering(linkage="ward",n_clusters=5)
agglomerative_clusters = agglo_model.fit_predict(customer_df)
# GaussianMixture algorithm
GaussianMixture_model = GaussianMixture(n_components=5)
gmm_clusters = GaussianMixture_model.fit_predict(customer_df)

# Now, calculate silhouette score to determine best clustering model
def silhouette_method(df,algo,y_pred):
  print('=================================================================================')
  print('Clustering ',algo," : silhouette score : ",silhouette_score(df,y_pred) )
  if algo == ' : GaussianMixture':
    print('=================================================================================')

silhouette_method(customer_df,' : KMeans',kmeans_clusters)
silhouette_method(customer_df,' : Agglomerative',agglomerative_clusters)
silhouette_method(customer_df,' : GaussianMixture',gmm_clusters)


## Visualize in 3D 
customer_df["label"] = kmeans_clusters
 
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
colors = ['blue','red','green','orange','purple']

for i in range(5):
    ax.scatter(customer_df.Age[customer_df.label == i], customer_df["Annual Income (k$)"][customer_df.label == i], customer_df["Spending Score (1-100)"][customer_df.label == i], c=colors[i], s=60)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()
