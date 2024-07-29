#!/usr/bin/env python
# coding: utf-8

# ## Upload and Transform Data


#Install API
#!pip install nfl_data_py

#Import Packages
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import nfl_data_py as pgskn


#Create variable to pull the games and outcomes of each event

df_sched = pgskn.import_schedules([2023])



#Create point difference variable for games 
df_sched['point difference'] = df_sched['home_score']-df_sched['away_score']


#determine the home advantage impact on the team
df_sched['home_win'] = np.where(df_sched['point difference'] > 0, 1, 0)
df_sched['home_loss'] = np.where(df_sched['point difference'] < 0, 1, 0)

#Create dummy data of home and away teams
df_home = pd.get_dummies(df_sched['home_team'], dtype=np.int64)
df_away = pd.get_dummies(df_sched['away_team'], dtype=np.int64)

#Create table giving away team -1 and home team 1
df_model = df_home.sub(df_away)
df_model['point difference'] = df_sched['point difference']


# ## Build Regression Model


#Pull Ridge regression down to incorporate to training data set

lr = Ridge(alpha = 1.0)
x_train = df_model.drop(['point difference'], axis = 1)
y_train = df_model['point difference']

#Eliminate NaN Values
x = x_train.fillna(x_train.interpolate())
y = y_train.fillna(y_train.interpolate())

#lr.fit(x,y)

#Create Ratings Variables to Determine the current strength of team based on performance to this point
#Larger(more positive) coefficients = Stronger performance 
df_ratings = pd.DataFrame(data={'Team': x.columns, 'Strength Rating': lr.coef_})
df_ratings

