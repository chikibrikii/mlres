
# coding: utf-8

# In[ ]:

import os
import glob
import matplotlib as plt
from matplotlib import style
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing, cross_validation
get_ipython().magic(u'matplotlib inline')
style.use('ggplot')


# source directory path.
src_dir = r"C:\Users\chsoon\Desktop\airline_case\Monthly Data"

# create a list containing each file directory path.
all_files = glob.glob(src_dir + "/*.csv")

# concatenate all files into single dataframe.
main_df = pd.DataFrame()
main_list = []
for file in all_files:
    df = pd.read_csv(file, index_col=None, header=0)
    main_list.append(df)
main_df = pd.concat(main_list)

# lookup tables' paths.
airport_id_dir = r"C:\Users\chsoon\Desktop\airline_case\Lookup Tables\L_AIRPORT_ID.csv"
carrier_hist_dir = r"C:\Users\chsoon\Desktop\airline_case\Lookup Tables\L_CARRIER_HISTORY.csv"

# create respective lookup table dataframes.
airport_id_df = pd.read_csv(airport_id_dir)
carrier_hist_df = pd.read_csv(carrier_hist_dir)


# In[ ]:

# drop all date and id fields.
main_df.drop(['FL_DATE',
              'AIRLINE_ID', 
              'DEST_AIRPORT_ID',
              'DEST_AIRPORT_SEQ_ID',
              'DEST_CITY_MARKET_ID',
              'ORIGIN_AIRPORT_ID',
              'ORIGIN_AIRPORT_SEQ_ID',
              'ORIGIN_CITY_MARKET_ID'], 1, inplace=True)

# convert all values to numeric type, if not int64 or float.
main_df.convert_objects(convert_numeric=True)
main_df.fillna(0, inplace=True)

# function to handle non-numeric data.
def handle_non_numeric_data(df):
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}  # empty dict of text and unique digits.
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != float:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1        
            df[column] = list(map(convert_to_int, df[column]))
        
    return df

main_df = handle_non_numeric_data(main_df)
# preprocess and scale data frame, but this changes the object type to non-pandas.
# main_df = preprocessing.scale(main_df)  


# In[ ]:

# predict the cluster assignment for each row in main data frame, writing to new CLUSTER column.
clf = KMeans(n_clusters=3)
main_df["CLUSTER"] = clf.fit_predict(main_df)


# In[ ]:

# PCA to create 2-D chart space.
pca = PCA(n_components=2)
main_df['x'] = pca.fit_transform(main_df)[:,0]
main_df['y'] = pca.fit_transform(main_df)[:,1]
main_df = main_df.reset_index()

