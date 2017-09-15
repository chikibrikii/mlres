# coding: utf-8
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing, cross_validation
get_ipython().magic(u'matplotlib inline')


# source file directory.
src_dir = r"C:\Users\chsoon\Desktop\airline_case\Monthly Data"

# create list of all files in source directory.
all_files = glob.glob(src_dir + "/*.csv")

# concatenate all files into single dataframe.
main_df = pd.DataFrame()
main_list = []
for file in all_files:
    df = pd.read_csv(file, index_col=None, header=0)
    main_list.append(df)
main_df = pd.concat(main_list)


airport_id_dir = r"C:\Users\chsoon\Desktop\airline_case\Lookup Tables\L_AIRPORT_ID.csv"
carrier_hist_dir = r"C:\Users\chsoon\Desktop\airline_case\Lookup Tables\L_CARRIER_HISTORY.csv"

# create lookup table dataframes.
airport_id_df = pd.read_csv(airport_id_dir)
carrier_hist_df = pd.read_csv(carrier_hist_dir)

# merge all dataframes on airport id, unique carrier name. 
airport_id_df.rename(columns={'Code':'ORIGIN_AIRPORT_ID', 'Description':'AIRPORT_NAME'}, inplace=True)
main_df = pd.merge(main_df, airport_id_df, on='ORIGIN_AIRPORT_ID')
carrier_hist_df.rename(columns={'Code':'UNIQUE_CARRIER', 'Description':'CARRIER_HIST'}, inplace=True)
main_df = pd.merge(main_df, carrier_hist_df, on='UNIQUE_CARRIER')

# What carrier has flown the third most number of flights? ANSWER: Atlantic Southeast Airlines, 686,021 flights.
carrier = main_df.groupby('CARRIER_HIST').FLIGHTS.value_counts().sort_values(ascending=False).iloc[[2]]

# What is the 15th most flown route? ANSWER: Nashville, TN - Chicago, IL, 13,818 flights.
route_df = main_df[['ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'FLIGHTS']]
route_df['ROUTE'] = route_df['ORIGIN_CITY_NAME'] + ' - ' + route_df['DEST_CITY_NAME']
route = route_df.groupby('ROUTE').FLIGHTS.value_counts().sort_values(ascending=False).iloc[[14]]

# What is the second most popular day of the week to travel? Why? ANSWER: Wednesday, 1,167,033 flights.  
main_df['MY_DATE'] = pd.to_datetime(main_df['FL_DATE'])
main_df['DAY'] = main_df['MY_DATE'].dt.weekday_name
day = main_df.groupby('DAY').FLIGHTS.value_counts().sort_values(ascending=False).iloc[[1]]

# What airport has the tenth most delays? ANSWER: New York, NY: LaGuardia, 168,083 delays.  
main_df['ACTUAL_DEP_DELAY'] = main_df['DEP_DELAY'] > 0
main_df['ACTUAL_ARR_DELAY'] = main_df['ARR_DELAY'] > 0
delays = main_df.groupby('AIRPORT_NAME').FLIGHTS.value_counts(('ACTUAL_DEP_DELAY' == True) & ('ACTUAL_ARR_DELAY' == True)).sort_values(ascending=False).iloc[[9]]

# list all column values.
main_df.columns.values.tolist()