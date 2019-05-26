#%%
# import libraries
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import random


#%%
# read in the listings
listings_df = pd.read_csv("./Data/listings.csv")
listings_df.head()


#%%
# read in the reviews
reviews_df = pd.read_csv("./Data/reviews.csv")
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
# distrbution of useage
df['useage'].hist(bins=100)



#%%
# How much income was generated 
df['income'] = df['price'] * df['useage']/100 * df['availability_365']
print(df['income'].sum())

df['income'].hist(bins=100);
print(df['income'].describe())




#%%
# monte-carlo simulation of income generated
def get_rand_number():
    """
    This function generates a random number between 1 and 100
    Return:
    - Random number (float)
    """
    roll = random.randint(1,100)
    roll = roll/100
    return roll

def monte_carlo_sim(df, prob = 0.2, number_sim = 1000):
    """
    This function generates a random number between 1 and 100
    Return:
    - Random number (int)
    """
    x = 0
    final_income = []
    while x < number_sim:  
        income_iter = 0      
        for index, row in df.iterrows():
            result = get_rand_number()
            if(result <= prob):
                income = row['price'] * row['availability_365'] * prob
                income_iter += income
        final_income.append(income_iter)
        x+=1 # increase the count
    return final_income


sim_outputs = monte_carlo_sim(df, prob = 0.25, number_sim = 10)
print(sim_outputs)


		

