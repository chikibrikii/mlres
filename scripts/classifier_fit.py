
# coding: utf-8

# In[1]:

import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing, cross_validation
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')




# source directory path.
src_dir = r"C:\Users\chsoon\Desktop\airline_case\Monthly Data"

# create list of each file directory.
all_files = glob.glob(src_dir + "/*.csv")

# concatenate all files into single dataframe.
main_df = pd.DataFrame()
main_list = []
for file in all_files:
    df = pd.read_csv(file, index_col=None, header=0)
    main_list.append(df)
main_df = pd.concat(main_list)

# lookup tables paths.
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

# create new columns
# main_df["AIR_TIME_HOURS"] = main_df["AIR_TIME"] / 60
main_df['ACTUAL_DEP_DELAY'] = main_df['DEP_DELAY'] > 0
main_df['ACTUAL_ARR_DELAY'] = main_df['ARR_DELAY'] > 0

main_df.columns.values.tolist()


# In[2]:

flights_series = main_df.groupby('CARRIER_HIST',as_index=True).FLIGHTS.value_counts().sort_values(ascending=False)

label = []
count = []
for i in flights_series.index:
    label.append(i[0])
    count.append(flights_series[i])
    
sub_df = pd.DataFrame({'CARRIER_HIST':label, 'FLIGHTS':count})
print sub_d


# In[ ]:

# main_df[main_df.CANCELLED == 1].sum().groupby('CARRIER_HIST').sort_values(ascending=False)  


# In[14]:

airtime_series = main_df.groupby('CARRIER_HIST', as_index=True).AIR_TIME.sum().sort_values(ascending=False)
label = []
count = []
for i in airtime_series.index:
    label.append(i[0])
    count.append(airtime_series[i])
    
bub_df = pd.DataFrame({'CARRIER_HIST':label, 'AIR_TIME':count})

combine_df = pd.merge(sub_df, bub_df, on='CARRIER_HIST')
print label


# In[8]:

# drop all date and id fields.
main_df.drop(['FL_DATE',
              'AIRLINE_ID',
              'FL_NUMBER',
              'DEST_AIRPORT_ID',
              'DEST_AIRPORT_SEQ_ID',
              'DEST_CITY_MARKET_ID',
              'ORIGIN_AIRPORT_ID',
              'ORIGIN_AIRPORT_SEQ_ID',
              'ORIGIN_CITY_MARKET_ID'], 1, inplace=True)


# In[ ]:

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


# In[ ]:

plt.scatter(main_df["DISTANCE"], main_df["ARR_DELAY"])
plt.ylabel('Arrival Delays')
plt.xlabel('Distance (miles)')


# In[ ]:

plt.scatter(main_df["DISTANCE"], main_df["AIR_TIME_MINS"])
plt.xlabel('Distance (miles)')
plt.ylabel('Airtime (mins)')


# In[ ]:

main_df["AIR_TIME_MINS"] = main_df["AIR_TIME"] / 60
plt.scatter(main_df["DISTANCE"], main_df["AIR_TIME_MINS"])
plt.ylabel('Distance')
plt.xlabel('Airtime (mins)')


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


# In[ ]:

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


# In[ ]:

main_df = handle_non_numeric_data(main_df)


# In[ ]:

main_df.head(3)


# In[ ]:

plt.scatter(main_df["DISTANCE"], main_df["DEP_DELAY"])


# In[ ]:

# preprocess and scale main data frame.
main_df = preprocessing.scale(main_df)


# In[ ]:

# train classifier. Do not need cross validation, as it is clustering.
clf = KMeans(n_clusters=3)
clf.fit(main_df)


# In[ ]:

# define classifer.
clf = KMeans(n_clusters=4)
clf.fit()

#access some of the attributes after fitting.
centroids = clf.cluster_centers_
labels = clf.labels_


# In[ ]:

# run kMeans on main dataframe.


# feature extraction 


# split cluster data set into training, cross-validation and test.



# run SVM on each cluster.



# reference lookup tables and append descriptions for results. 

