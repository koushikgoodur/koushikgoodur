# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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
df = pd.read_csv('Bollywood songs.csv')
df = df.rename({'Unnamed: 0': 'UserID'}, axis=1)
df=df.fillna(0)
df = df.replace('10-Oct', np.NaN)
replace = df.fillna(df.mode().iloc[0], inplace=True)
df = df.replace('09-Oct', np.NaN)
replace = df.fillna(df.mode().iloc[0], inplace=True)
df["UserID"].replace({"8.8/10": "2421"}, inplace=True)
df=df.rename(columns={'Song-Name':'Song','Singer/Artists':'Singer','Album/Movie':'Movie','User-Rating':'Rating'})
df['Rating'] = df['Rating'].str.replace('/10', '')
df['UserID'] = pd.to_numeric(df['UserID'],errors = 'coerce')
df['Rating'] = pd.to_numeric(df['Rating'],errors = 'coerce')



var1 = st.sidebar.radio("New user?", ["Yes", "No"])



if var1 == "Yes":
    #Cold start code
    data1 =df.groupby('Song')['Rating'].sum()
    data12 = pd.DataFrame(data1)
    sorted_data12  = data12.sort_values("Rating", ascending= False)
    Sorted_data12_reset_index = sorted_data12.reset_index()

    st.header("Recommended list of songs is :")

    for i in range(20):
        st.write(i+1, Sorted_data12_reset_index["Song"][i])

else:
    
    
    song = df.pivot_table(index='UserID',columns='Song',values='Rating').reset_index(drop=True)

    
    # replacing NANs with 0
    song.fillna(0, inplace=True)

    #Calculating Cosine Similarity between Users

    user_sim = 1 - pairwise_distances( song.values,metric='cosine')

    # storing user similarity inside the dataframe 
    user_df = pd.DataFrame(user_sim)

    
    np.fill_diagonal(user_sim, 0)


    # function for giving the recommendation
    def give_reco(song):
        tem=list(user_df.sort_values([song],ascending=False).head(10).index)
        songs_list=[]
        for i in tem:
            songs_list=songs_list+list(df[df["UserID"]==i]["Song"])

        a = set(songs_list)-set(df[df["UserID"]==song]["Song"])
        
        st.header("Recommended list of songs is:")
        for item in a:
            st.write(item)

            
    give_reco(2000)
