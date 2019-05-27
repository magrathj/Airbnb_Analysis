#%%
# import libraries
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import random
from scipy.stats import expon
from scipy.stats import gamma
from numpy import array
import seaborn as sns
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import time
plotly.offline.init_notebook_mode()
sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(5,5)})

#%%
# read in the listings
listings_df = pd.read_csv("./Data/london_listings.csv")
listings_df.head()


#%%
# read in the reviews
reviews_df = pd.read_csv("./Data/london_reviews.csv")
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
reviews_df.head()

#%%
# filter dates 

start_date = '2018-01-01'
end_date = '2018-12-31'
mask = (reviews_df['date'] > start_date) & (reviews_df['date'] <= end_date)
reviews_df = reviews_df.loc[mask]
reviews_df = reviews_df.groupby('listing_id').count()
reviews_df.head()

#%%
# summary stats around each dataframe
print('--------------')
print('Review shape')
print(reviews_df.shape)

print('--------------')
print('Listing shape')
print(listings_df.shape)

print('--------------')
print('NANs in reviews')
print(reviews_df.isna().sum())

print('--------------')
print('NANs in listings')
print(listings_df.isna().sum())




#%%
# describe review
 
print(reviews_df.describe())



#%%
# describe listings
 
print(listings_df.describe())



#%%
# distrbution of prices
listings_df['price'].hist(bins=100);


#%%
# box plot of prices
boxplot = listings_df.boxplot(column=['price'])


#%%
# distrbution of prices in log transform
plt.hist(listings_df['price'], log=True) 



#%%
# distrbution of dates
reviews_df['date'].hist(bins=100);


#%%
# describe of dates
reviews_df['date'].describe()



#%%
# merge listings and reviews
df = pd.merge(listings_df, reviews_df, left_on='id', right_on='listing_id')
df.head()



#%%
# check each id is unique - one row per id
print(len(df['id'].unique().tolist()))
print(df['id'].count())



#%%
# describe merged dfs
 
print(df.describe())



#%%
# Percentage of dates compared to availablity
def check_dates(df):
    """
    This function checks if the number of dates the property was 
    used went over the availability. If so then it changes the availability 
    of the property to 365.
    Args:
    - df (pandas dataframe)
    Return:
    - returns column availability_365 results 
    """
    if (df['date'] > df['availability_365'] ):
        return 365
    elif (df['date'] <= df['availability_365'] ):
        return df['availability_365'] 

df = df[(df[['availability_365']] != 0).all(axis=1)]
df['availability_365'] = df.apply(check_dates, axis = 1)

df['useage'] = df['date']/df['availability_365'] * 100
df['useage'].head()







#%%
# describe useage of properties

print(df.useage.describe())

print('--------------')
print('NANs in df')
print(df.isna().sum())

print('--------------')
print('mean of usage')
print(df.useage.mean())

print('--------------')
print('median of usage')
print(df.useage.median())




#%%
# How much income was generated 
df['income'] = df['price'] * df['useage']/100 * df['availability_365']
print(df['income'].sum())

df['income'].hist(bins=100);
print(df['income'].describe())


#%%
# distrbution of useage
#df['useage'].hist(bins=100);
data =  df[df['useage'] < 60]
print(df[df['useage'] < 60].useage.median())
ax = sns.distplot(data['useage'],
                  kde=True,
                  bins=100,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Exponential Distribution', ylabel='Frequency')



#%%
# fitting a expontial distrbution to useage
data_expon = expon.rvs(scale=5.5, loc=0, size=1000)
data_expon.mean()

#%%
# fitting a expontial distrbution to useage
data_expon = expon.rvs(scale=5.5, loc=0, size=1000)
ax = sns.distplot(data_expon,
                  kde=True,
                  bins=100,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Exponential Distribution', ylabel='Frequency')


#%%
# fitting a gamma distrbution to useage
data_gamma = gamma.rvs(a=20, size=10000)
ax = sns.distplot(data_gamma,
                  kde=True,
                  bins=100,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Exponential Distribution', ylabel='Frequency')




#%%
# monte-carlo simulation functions
def get_rand_numbers(scale=5, size=1000):
    """
    This function generates a random number from a gamma distribution
    Return:
    - Random number (float)
    """
    roll = expon.rvs(scale=scale, loc=0, size=size)
    roll = roll/100
    return roll

def monte_carlo_sim(df, scale =4.9, number_sim = 100):
    """
    This function generates a monte-carlo simulation to determine the distribution of 
    overall incomes which are possible given a specific dataframe
    and gamma distribution
    Return:
    - final_income (array of int) each value represents the sum of the income from that simulation
    """
    x = 0
    final_income = []
    while x < number_sim:  
        sim_outputs = df['price'] * df['availability_365']
        y = np.array(get_rand_numbers(scale = scale, size=sim_outputs.shape[0]))
        income_iter = sum(sim_outputs.multiply(y, axis=0))	
        final_income.append(income_iter)
        x+=1 # increase the count
    return final_income


#%%
# monte-carlo simulation of income generated
def plot_simulation(df, scale = 5.2, number_sim = 1000):
    """
    This function generates a monte-carlo simulation to determine the distribution of 
    overall incomes which are possible given a specific dataframe
    and gamma distribution
    Return:
    - final_income (array of int) each value represents the sum of the income from that simulation
    """
    sim_outputs = monte_carlo_sim(df=df, scale=scale, number_sim = number_sim)
    data = array(sim_outputs)
    plt.xlim([min(data), max(data)])
    plt.hist(data, bins=100)
    plt.title('Random Gaussian data (fixed bin size)')
    plt.xlabel('variable X (bin size = 5)')
    plt.ylabel('count')
    plt.axvline(df['income'].sum(), color='r')
    plt.show()



plot_simulation(df , scale = 5.2, number_sim = 100000)






