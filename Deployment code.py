#!/usr/bin/env python
# coding: utf-8


# installing necessary packages
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation



st.title("Devotional Songs Recommender System")

# In[ ]:
# reading CSV file
df = pd.read_csv(r'C:\Users\koush\Bollywood songs.csv')
data = df.sample(2421)
users = list(data['UserID'].unique())

var1 = st.sidebar.radio("New user?", ["Yes", "No"])


if var1 == "Yes":
    #Cold start code
    data1 =data.groupby('Song')['Rating'].sum()
    data12 = pd.DataFrame(data1)
    sorted_data12  = data12.sort_values("Rating", ascending= False)
    Sorted_data12_reset_index = sorted_data12.reset_index()

    st.header("Recommended list of songs is :")

    for i in range(20):
        st.write(i+1, Sorted_data12_reset_index["Song"][i])

else:
    
    customer_id = st.selectbox("Please select the user-ID: ", users)
    # Creating pivot table
    song = df.pivot_table(index='UserID',
                                     columns='Song',
                                     values='Rating')

    # replacing NANs with 0
    user_df.fillna(0, inplace=True)

    #Calculating Cosine Similarity between Users

    user_sim = 1 - pairwise_distances( song.values,metric='cosine')

    # storing user similarity inside the dataframe 
    user_df = pd.DataFrame(user_sim)

    #Set the index and column names to user ids 
    user_df.index = df.UserID.unique()
    user_df.columns = df.UserID.unique()

    np.fill_diagonal(user_sim, 0)


    # function for giving the recommendation
    def give_reco(song):
        tem=list(user_df.sort_values([song],ascending=False).head(10).index)
        songs_list=[]
        for i in tem:
            songs_list=songs_list+list(data[data["UserId"]==i]["Song"])

        a = set(songs_list)-set(df[df["user_id"]==song]["Song"])
        
        st.header("Recommended list of songs is:")
        for item in a:
            st.write(item)
            
    give_reco(song)









