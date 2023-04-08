#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sb
import scipy.stats as stats
from pydataset import data
from sklearn.model_selection import train_test_split
import env


# In[3]:


# Zillow:

##################### *ACQUIRE* ##########################
url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow'

def get_zillow():
    """ This function pulls information from the mySQL zillow database and returns it as a
    pandas dataframe"""
    sql = """ select bedroomcnt, bathroomcnt,
calculatedfinishedsquarefeet, fips, lotsizesquarefeet,
 taxvaluedollarcnt, yearbuilt, assessmentyear, taxamount
 from properties_2017 where propertylandusetypeid = '261' limit 1500000 """
    df = pd.read_sql(sql, url)
    return df

#################### *PREPARE* ############################

def prep_zillow(df):
    """ This function prepares/cleans data from the zillow df for splitting"""
    
    # Drops null values from columns
    df= df.dropna()
    
    #Renames columns to something more visual appealing
    df = df.rename(columns= {'bedroomcnt': 'beds', 'bathroomcnt':'baths',
                        'taxamount':'tax_amt', 'lotsizesquarefeet':'lot_size',
                        'calculatedfinishedsquarefeet':'sq_ft',
                        'taxvaluedollarcnt':'tax_val','yearbuilt':'year',
                        })
    
    return df

################### *OUTLIERS* #########################
# brought to you by Madeleine
def banish_them(df, col_list, k=1.5):
    """This function removes outliers from the columns in the dataframe 
    , then returns the dataframe """
    
    #Quartiles
    q1, q3 = df[col].quantile([.25, .75])
    
    #Interquartile Range
    iq = q3 - q1
    
    #Sets Upper Bound
    up_bnd= q3 + k * iq
    
    #Sets lower bound
    lo_bnd = q1 - k * iq
    
    df= df[(df[col] > lo_bnd) & (df[col] < up_bnd)]
    
    return df
################### *SPLIT* ###################
def split_zillow(df):
    '''
    take in a DataFrame return train, validate, test split on zillow DataFrame.
    '''
# Reminder: I don't need to stratify in regression. I don't remember why, but Madeleine said 
# it
    train_val, test = train_test_split(df, test_size=.2, random_state=123)
    train, val = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    return train, val, test

################## *WRANGLE* ################
def wrangle_zillow():
    
    train, val, test = prep_zillow(get_zillow())
    
    return train, val, test


# In[4]:


# Student Grades:


import os

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_student_data():
    filename = "student_grades.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM student_grades', get_connection('school_sample'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df
    
def wrangle_grades():
    '''
    Read student_grades into a pandas DataFrame from mySQL,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    '''

    # Acquire data

    grades = get_student_data()

    # Replace white space values with NaN values.
    grades = grades.replace(r'^\s*$', np.nan, regex=True)

    # Drop all rows with NaN values.
    df = grades.dropna()

    # Convert all columns to int64 data types.
    df = df.astype('int')

    return df


# -------------Feature Engineering-----------------

def select_kbest(X, y, k=2):
    """ X: df of independent features
        y: target
        k: number of kbest features to select. defaulted to 2, but can be changed)"""
    
 # make
    kbest= SelectKBest(f_regression, k=k)
    
 # fit
    kbest.fit(X, y)
 # get support
    mask = kbest.get_support()
    return X.columns[mask]





