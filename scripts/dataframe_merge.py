
# coding: utf-8

# In[1]:

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


# In[2]:

# source directory path.
src_dir = r"C:\Users\chsoon\Desktop\airline_case\Monthly Data"

# define paths to each file in Monthly Data directory.
# csv_1 = r"C:\Users\chsoon\Desktop\airline_case\Monthly Data\1.csv"
# csv_2 = r"C:\Users\chsoon\Desktop\airline_case\Monthly Data\2.csv"
# csv_3 = r"C:\Users\chsoon\Desktop\airline_case\Monthly Data\3.csv"

# create list of each file directory.
all_files = glob.glob(src_dir + "/*.csv")
# print all_files

# concatenate all files into single dataframe.
main_df = pd.DataFrame()
main_list = []

for file in all_files:
    df = pd.read_csv(file, index_col=None, header=0)
    main_list.append(df)
main_df = pd.concat(main_list)

# count number of rows in main dataframe.
# main_df.shape[0]
# check dimensions of main dataframe.
main_df.shape

# lookup tables paths.
airport_id_dir = r"C:\Users\chsoon\Desktop\airline_case\Lookup Tables\L_AIRPORT_ID.csv"
carrier_hist_dir = r"C:\Users\chsoon\Desktop\airline_case\Lookup Tables\L_CARRIER_HISTORY.csv"

# create lookup table dataframes.
airport_id_df = pd.read_csv(airport_id_dir)
carrier_hist_df = pd.read_csv(carrier_hist_dir)


# In[10]:

airport_id_df.rename(columns={'Code':'ORIGIN_AIRPORT_ID', 'Description':'AIRPORT_NAME'}, inplace=True)
main_df = pd.merge(main_df, airport_id_df, on = 'ORIGIN_AIRPORT_ID')

