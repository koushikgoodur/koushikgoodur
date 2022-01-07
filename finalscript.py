# -*- coding: utf-8 -*-
"""finalscript.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18Gr2K1N75xaZhNkidtbx1yLkKnmLUT_I

# Importing The Libraries
"""

import pandas as pd
import numpy as np

"""# Import the Dataset """

df = pd.read_csv('devoltional song2.csv')
df

"""# Drop the Columns"""

df = df.drop(['Release Date','Added At','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo','Time Signature','Genres','Danceability','Energy','Artist IDs'],axis=1)

df

"""# Column Rename"""

data = df.rename(columns={'Spotify ID':'spotify_id','Artist IDs':'artist_id','Track Name':'track_name','Album Name':'album_name','Artist Name(s)':'singers_name','Duration (ms)':'duration_ms'})

data

"""# Import The Dataset 1"""

df1 = pd.read_csv('devotional song 1 .csv')
df1

"""# Drop the columns"""

df1 = df1.drop(['Artist URI(s)','Album URI','Album Artist URI(s)','Album Artist Name(s)','Album Release Date','Album Image URL','Disc Number','Track Number','Track Preview URL','Explicit','Added At'],axis=1)

df1

"""# Rename the columns"""

data1 = df1.rename(columns={'Track URI':'spotify_id','Track Name':'track_name','Artist Name(s)':'singers_name','Track Duration (ms)':'duration_ms','Album Name':'album_name'})

data1

"""# Import The Dataset 2"""

df2 = pd.read_csv('devotional_songs.csv')
df2

"""# Drop the columns"""

df2 = df2.drop(['Release Date','Added At','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo','Time Signature','Genres','Danceability','Energy','Artist IDs'],axis=1)

df2

"""# Rename the column"""

data2 = df2.rename(columns={'Spotify ID':'spotify_id','Artist IDs':'artist_id','Track Name':'track_name','Album Name':'album_name','Artist Name(s)':'singers_name','Duration (ms)':'duration_ms'})

data2

"""# Now the Merge the dataset dataset1 dataset2"""

devotional_data = pd.concat([data,data1,data2],axis=0).reset_index()

devotional_data

"""# Data preprocessing"""

devotional_data["duration"]=devotional_data["duration_ms"].apply(lambda x:round(x/1000)%60)
devotional_data.drop(["duration_ms"],inplace=True,axis=1)

"""# now merge the column track_name and singers_name and the new column name is song"""

devotional_data['song'] = devotional_data['track_name']+'-'+devotional_data['singers_name']
devotional_data.head()

sort_data = devotional_data.sort_values('Popularity',ascending= False).head(10)

sort_data

sort_data1 = devotional_data.sort_values('Popularity',ascending=True).head(10)

sort_data1

devotional_data.describe(include='all').transpose()

"""# Most popular song by Populrity """

most_popular = devotional_data.query('Popularity>40', inplace=False).sort_values('Popularity',ascending=False)
most_popular.head(10)

devotional_data.groupby('album_name')['Popularity'].mean()

devotional_data.groupby('track_name')['Popularity'].mean()

devotional_data.groupby('album_name')['duration'].mean()

devotional_data.groupby('track_name')['duration'].mean()

"""# Pivot table"""

devotional_song = pd.pivot_table(data=devotional_data,values='Popularity',index='spotify_id',columns='song').fillna(0)
devotional_song

X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.20,random_state=20)

X_train.shape,y_train.shape

X_test.shape,y_test.shape

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(devotional_data)
    wcss.append(kmeans.inertia_)
    print(i,wcss)
    
    

plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

cluster_new = KMeans(n_clusters=6,random_state=20)
cluster_new.fit(devotional_data)

cluster_new.labels_

cluster_new.cluster_centers_

devotional_data['cluster_new'] = cluster_new.labels_

devotional_data

devotional_data.groupby('cluster_new').agg(['mean'])

devotional_data

devotional_data[devotional_data['cluster_new']==5]

"""# Now we use to UBCF - User Based Collabrated Filtering"""

devotional_transpose = devotional_song.T
devotional_transpose

devotional_correation = devotional_transpose.corr().round(2).fillna(0)
devotional_correation

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation

devotional_song.values

user_to_user = pairwise_distances(X=devotional_song.values,metric='euclidean')
user_to_user

user_to_user_song = pd.DataFrame(data=user_to_user,index=devotional_song.index,columns=devotional_song.index)
user_to_user_song

sns.heatmap(data=user_to_user_song)

user_to_user1 = 1-pairwise_distances(X=devotional_song,metric='euclidean')
user_to_user1

user_to_user_song1=pd.DataFrame(data=user_to_user1,index=devotional_song.index,columns=devotional_song.index)
user_to_user_song1

sns.heatmap(data=user_to_user_song1)

user_to_user_song.idxmax()

user_to_user_song1.idxmax()

user_to_user_cosine=1-pairwise_distances(X=devotional_song.values,metric='cosine')
user_to_user_cosine

user_to_user_cosine1 = pd.DataFrame(data=user_to_user_cosine,index=devotional_song.index,columns=devotional_song.index)
user_to_user_cosine1

sns.heatmap(data=user_to_user_cosine1)

user_to_user_cosine1.idxmax()

devotional_data[(devotional_data['spotify_id']=='01Tul0EVB3EmpepTYBcepp') | (devotional_data['spotify_id']=='5QtQFSdyZEl0w4iDxAyv76')]































