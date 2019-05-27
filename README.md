# Using Monte Carlo Simulations to find out how much money Airbnb makes in different Cities

Airbnb provides a whole host of really useful data for free. So why not do something interesting with it to try to answer some real world business problems that I see crop up quite often at my work?

First one being: how much income does a specific area generate and how does that compare to other areas? More specifically where should we spend more of our advertisement budget to generate more returns — which geographic area should we focus on and how much return can we make?


## Five Questions that the analysis in the .ipynb files are trying to answer


### How much income do properties in London generate in terms of income for Airbnb?

### How much income do properties in Cape Town generate in terms of income for Airbnb?

### How much income could Airbnb earn if it increased its average percentage of days a property is rented for compared to the days its available for in London?

### Which location generates more income for Airbnb - Cape Town, SA or London, UK?

### Where should Airbnb spend its advertising budget - in Cape Town, SA or London, UK?


## Outcomes

In the results I go through Airbnb's interesting datasets and estimated how much income they generate in different cities. I then use Monte Carlo simulations to estimate the impact of an increase of 5% to the mean for estimated occupancy and predicted the resulting income that would generate (£18.7m). The power of this simple techniques is that you can start to quantify some of the uncertainty associated with decision making process in the workplace and allow decision makers to make informed calls on what to do next.

## Blog Link to Medium article
https://medium.com/@magrathj/using-monte-carlo-simulations-to-find-out-how-much-money-airbnb-makes-in-different-cities-a8bdf19a4a58


## Libraries used with 3.7.1 Python enviroment
### import pandas as pd
### import matplotlib.pyplot as plt 
### import numpy as np
### import random
### from scipy.stats import expon
### from scipy.stats import gamma
### from numpy import array
### import seaborn as sns
### import plotly
### import plotly.plotly as py
### import plotly.graph_objs as go
### import time


## Data sources

### London
http://data.insideairbnb.com/united-kingdom/england/london/2019-05-05/visualisations/listings.csv
http://data.insideairbnb.com/united-kingdom/england/london/2019-05-05/visualisations/reviews.csv

### Cape-Town
http://data.insideairbnb.com/south-africa/wc/cape-town/2019-04-18/visualisations/listings.csv
http://data.insideairbnb.com/south-africa/wc/cape-town/2019-04-18/visualisations/reviews.csv






## Reference
https://www.datacamp.com/community/tutorials/probability-distributions-python
https://pythonprogramming.net/monte-carlo-simulator-python/
