import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cluster

# Source of Picture
df = pd.read_csv("https://stanford.edu/~lcambier/pc/old_faithful.csv")
df.head(5)

#Isolate a portion of the picture for modeling
X = df.loc[:,['waiting','eruptions']].to_numpy()


# This scales the data around 0 with a std of 1
from sklearn import preprocessing
X_scaled = preprocessing.scale(X)

# plt.scatter(X_scaled[:,0], X_scaled[:,1])
# plt.xlabel('Waiting')
# plt.ylabel('Eruption time')
# plt.title('Old Faithful Geyser (scaled)')
# plt.show()

#Perform Kmeans on X_scaled
model = sklearn.cluster.KMeans(n_clusters=2)
kmeans = model.fit(X_scaled)

#Plot centers of data
Xcenters = kmeans.cluster_centers_
# plt.scatter(X_scaled[:,0], X_scaled[:,1])
# plt.scatter(Xcenters[:,0], Xcenters[:,1], s=400, marker='*', c='black')
# plt.title('Old Faithful Geyser (scaled)')
# plt.show()

# This generate 1000 random points over [40,100] x [1.5,5.5] (in original units)
Xpred = np.random.uniform(low=(40,1.5),high=(100,5.5),size=(1000,2))


#Predict the cluster using Kmeans prediction
colors_map = {0:'red',1:'blue'}
Xpred_scaled = preprocessing.scale(Xpred)
ypred = kmeans.predict(Xpred_scaled)
plt.scatter(Xpred[:,0], Xpred[:,1], c=[colors_map[i] for i in ypred])
plt.scatter(X[:,0], X[:,1], c='black')
plt.xlabel('Waiting')
plt.ylabel('Eruption time')
plt.title('Old Faithful Geyser')
plt.show()
