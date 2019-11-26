import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.chdir('C:/Users/CosmicDust/Documents/Statistics/Assignment 5')

# Reading the dataset
df = pd.read_excel('DEN 2012 - Jun 17-1.xlsx')
df.head()
df.describe()
df.info()

#---------------------------- DATA PREPROCESSING ------------------------------

import datetime

# Convert datetime format to String
df['Month and Year'] = df['Month and Year'].dt.strftime('%Y-%m-%d')

# Extract the year out of Month of Year column. (The format of actual data was incorrect)
df['Year'] = df['Month and Year'].apply(lambda date: date.split('-')[2])
#df['Year'] = pd.to_numeric(df['Year'])

# Drop Month and Year column because it is redundant
df = df.drop(['Month and Year'],axis=1)

# Create a new column 'Month and Year' by merging Month and Year columns
df['Month and Year'] = df[['Month', 'Year']].apply(lambda x: ''.join(x), axis=1)
df.head()

#------------------------- EXPLORATORY DATA ANALYSIS --------------------------
sns.set_style('darkgrid')

# Checking if Cannbis column plays an important factor?
plt.figure(figsize=(12,6)),sns.barplot(x='Month',y='Ground',data=df,hue='Cannabis?')
plt.figure(figsize=(12,6)),sns.barplot(x='Month',y='Parking',data=df,hue='Cannabis?')
plt.figure(figsize=(12,6)),sns.barplot(x='Month',y='Rental Car',data=df,hue='Cannabis?')
plt.figure(figsize=(12,6)),sns.barplot(x='Month',y='Concession',data=df,hue='Cannabis?')


# Scatterplots
plt.figure(figsize=(12,6)),sns.scatterplot(x='Enplaned',y='Concession',data=df)
plt.figure(figsize=(12,6)),sns.scatterplot(x='Enplaned',y='Parking',data=df)
plt.figure(figsize=(12,6)),sns.scatterplot(x='Transfer',y='Rental Car',data=df)
plt.figure(figsize=(12,6)),sns.scatterplot(x='Enplaned',y='Ground',data=df)

# Lineplots
plt.figure(figsize=(35,6)),sns.lineplot(x='Month and Year',y='Ground',data=df,sort=False)
plt.figure(figsize=(35,6)),sns.lineplot(x='Month and Year',y='Parking',data=df,sort=False)
plt.figure(figsize=(35,6)),sns.lineplot(x='Month and Year',y='Rental Car',data=df,sort=False)
plt.figure(figsize=(35,6)),sns.lineplot(x='Month and Year',y='Concession',data=df,sort=False)

# Histograms
sns.distplot(df['Enplaned'],bins=25,kde=False)
sns.distplot(df['Deplaned'],bins=25,kde=False)
sns.distplot(df['Transfer'],bins=25,kde=False)
sns.distplot(df['Originating'],bins=25,kde=False)
sns.distplot(df['Destination'],bins=25,kde=False)

# Linear Model graphs
sns.lmplot(x='Destination',y='Ground',data=df,col='Year')
sns.lmplot(x='Month and Year',y='Rental Car',data=df)
sns.lmplot(x='Month and Year',y='Rental Car',data=df)
sns.lmplot(x='Month and Year',y='Ground',data=df)

# Corelation Matrix
df.corr()
cmap = sns.diverging_palette(10, 10, as_cmap=True)
plt.figure(figsize=(14,10)),sns.heatmap(df.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)


df.columns

#-------------------------------- DATA CLEANING -------------------------------

# Set 'Month and Year' column as the index of dataframe
df = df.set_index('Month and Year')

# Function to replace negative values with positives
def fix_negatives(values):
    if values < 0:
        return abs(values)
    else:
        return values

# Applying above function to 'Concession' column to convert negative value to positive
df['Concession'] = df['Concession'].apply(fix_negatives)


# Substitute the missing value in 'Parking' using moving average of data in May each year
df['Parking']['May15'] = round(((df['Parking']['May12']*1) + 
                                (df['Parking']['May13']*2) + 
                                (df['Parking']['May14']*3) + 
                                (df['Parking']['May16']*4) + 
                                (df['Parking']['May17']*5)) / (1+2+3+4+5))

plt.figure(figsize=(20,10)),sns.barplot(x='Month',y='Parking',data=df, hue='Year')


# Function to change months into numeric format
def convert_month(cols):
    Month = cols[0]
    
    if Month == 'Jan':
        return 1
    elif Month == 'Feb':
        return 2
    elif Month == 'Mar':
        return 3
    elif Month == 'Apr':
        return 4
    elif Month == 'May':
        return 5
    elif Month == 'Jun':
        return 6
    elif Month == 'Jul':
        return 7
    elif Month == 'Aug':
        return 8
    elif Month == 'Sep':
        return 9
    elif Month == 'Oct':
        return 10
    elif Month == 'Nov':
        return 11
    else:
        return 12

# Apply the function
df['Month'] = df[['Month']].apply(convert_month,axis=1)
df.head()


df.to_excel('DIA-CleanData.xlsx', sheet_name='sheet1', index=False)


# WE DID NOT USE THESE MODELS, JUST TRIED TO MAKE IT IN PYTHONS AS WELL...

#------------------------------------------------------------------------------
#------------------------------------MODELS------------------------------------
#------------------------------------------------------------------------------

# Create Dataset from which we will subset dataset for each model
df.loc[:,['Month', 'Enplaned', 'Deplaned', 'Transfer', 'Originating','Destination', 'Concession', 'Parking', 'Rental Car', 'Ground','Origin + Destin']]

# --------------------------------- Concession --------------------------------

concession_X = df.loc[:,['Month', 'Enplaned', 'Deplaned', 'Transfer', 'Originating','Destination', 'Concession', 'Parking', 'Rental Car', 'Ground','Origin + Destin']]

# Multiplying least corelated features of Concession with it to increase their corelation
concession_X['Transfer'] = concession_X['Transfer']*concession_X['Concession']
concession_X['Month'] = concession_X['Month']*concession_X['Concession'].transform(func='square')

# Transformations
concession_X['Originating'] = concession_X['Originating']/concession_X['Rental Car']
concession_X['Ground'] = concession_X['Ground']/concession_X['Parking']


# Heatmap to check correlation of X. Using this, we can decide which features to keep in training and testing dataset
plt.figure(figsize=(14,10)),sns.heatmap(concession_X.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)

# Storing only needed features
concession_X = concession_X.loc[:,['Month','Transfer', 'Originating', 'Parking', 'Ground']]
concession_X = concession_X.transform(func='square')
concession_y = df.loc[:,['Concession']]


concession_X.corr()
plt.figure(figsize=(14,10)),sns.heatmap(concession_X.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)

# Split TRAIN and TEST data
concession_X_train = concession_X[:'Feb17']
concession_X_test = concession_X['Mar17':]
concession_y_train = concession_y[:'Feb17']
concession_y_test = concession_y['Mar17':]

# Check t-test
import scipy.stats as stats
stats.ttest_1samp(a = concession_X_train, popmean = concession_X.mean())

# Run Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(concession_X_train,concession_y_train)

# Predict
predict_concession = lm.predict(concession_X_test)


# Evaluation


print(f'Coefficients: {lm.coef_}')
print(f'Intercept: {lm.intercept_}')
print(f'R^2 score: {lm.score(concession_X, concession_y)}')

import statsmodels.api as sm

X_constant = sm.add_constant(concession_X)
lm = sm.OLS(concession_y,X_constant).fit()
lm.summary()



import statsmodels.stats.api as sms

sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)

def linearity_test(model, y):
    '''
    Function for visually inspecting the assumption of linearity in a linear regression model.
    It plots observed vs. predicted values and residuals vs. predicted values.
    
    Args:
    * model - fitted OLS model from statsmodels
    * y - observed values
    '''
    fitted_vals = model.predict()
    resids = model.resid

    fig, ax = plt.subplots(1,2)
    
    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
    ax[0].set(xlabel='Predicted', ylabel='Observed')

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
    
linearity_test(lm, concession_y)  

lm.resid.mean()  
    




import statsmodels.stats.api as sms

sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)

def homoscedasticity_test(model):
    '''
    Function for testing the homoscedasticity of residuals in a linear regression model.
    It plots residuals and standardized residuals vs. fitted values and runs Breusch-Pagan and Goldfeld-Quandt tests.
    
    Args:
    * model - fitted OLS model from statsmodels
    '''
    fitted_vals = model.predict()
    resids = model.resid
    resids_standardized = model.get_influence().resid_studentized_internal

    fig, ax = plt.subplots(1,2)

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Residuals vs Fitted', fontsize=16)
    ax[0].set(xlabel='Fitted Values', ylabel='Residuals')

    sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Scale-Location', fontsize=16)
    ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')

    bp_test = pd.DataFrame(sms.het_breuschpagan(resids, model.model.exog), 
                           columns=['value'],
                           index=['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'])

    gq_test = pd.DataFrame(sms.het_goldfeldquandt(resids, model.model.exog)[:-1],
                           columns=['value'],
                           index=['F statistic', 'p-value'])

    print('\n Breusch-Pagan test ----')
    print(bp_test)
    print('\n Goldfeld-Quandt test ----')
    print(gq_test)
    print('\n Residuals plots ----')

homoscedasticity_test(lm)





# ---------------------------------- Parking ----------------------------------

parking_X = df.loc[:,['Month', 'Enplaned', 'Deplaned', 'Transfer', 'Originating','Destination', 'Concession', 'Parking', 'Rental Car', 'Ground','Origin + Destin']]
parking_X.columns

# Heatmap to check correlation of X. Using this, we can decide which features to keep in training and testing dataset
plt.figure(figsize=(14,10)),sns.heatmap(parking_X.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)

# Multiplying least corelated features of Parking with square of it to increase their corelation
parking_X['Transfer'] = parking_X['Transfer']*parking_X['Parking'].transform(func='square')
parking_X['Month'] = parking_X['Month']*parking_X['Parking'].transform(func='square')
plt.figure(figsize=(14,10)),sns.heatmap(parking_X.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)

# Transformations
parking_X['Originating'] = parking_X['Originating']/parking_X['Rental Car'].transform(func='sqrt')
parking_X['Concession'] = parking_X['Concession']/parking_X['Origin + Destin'].transform(func='sqrt')
parking_X['Transfer'] = parking_X['Transfer']/parking_X['Destination'].transform(func='sqrt')


# Storing only needed features
parking_X = parking_X.loc[:,['Month','Transfer', 'Originating', 'Concession', 'Ground']]
parking_X = parking_X.transform(func='square')
parking_y = df.loc[:,['Parking']]


parking_X.corr()
plt.figure(figsize=(14,10)),sns.heatmap(parking_X.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)

# Split TRAIN and TEST data
parking_X_train = parking_X[:'Feb17']
parking_X_test = parking_X['Mar17':]
parking_y_train = parking_y[:'Feb17']
parking_y_test = parking_y['Mar17':]

# Check t-test
stats.ttest_1samp(a = parking_X_train, popmean = parking_X.mean())

# Run Linear Regression
lm = LinearRegression()

lm.fit(parking_X_train,parking_y_train)

# Predict
predict_parking = lm.predict(parking_X_test)

# Check R^2
lm.score(parking_X,parking_y)

# Check Intercept
lm.intercept_

# Check p-value again
stats.mstats.linregress(parking_y_test,predict_parking)

# ------------------------------- Rental Car ----------------------------------

rentalCar_X = df.loc[:,['Month', 'Enplaned', 'Deplaned', 'Transfer', 'Originating','Destination', 'Concession', 'Parking', 'Rental Car', 'Ground','Origin + Destin']]


# Heatmap to check correlation of X. Using this, we can decide which features to keep in training and testing dataset
plt.figure(figsize=(14,10)),sns.heatmap(rentalCar_X.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)

# Multiplying least corelated features of Rental Car with it to increase their corelation
rentalCar_X['Transfer'] = rentalCar_X['Transfer']*rentalCar_X['Rental Car']
rentalCar_X['Month'] = rentalCar_X['Month']*rentalCar_X['Rental Car'].transform(func='square')
plt.figure(figsize=(14,10)),sns.heatmap(rentalCar_X.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)

# Transformations
rentalCar_X['Transfer'] = rentalCar_X['Transfer'].transform(func='square')
rentalCar_X['Destination'] = rentalCar_X['Destination']/rentalCar_X['Originating']
rentalCar_X['Concession'] = rentalCar_X['Concession']/rentalCar_X['Enplaned']
rentalCar_X['Parking'] = rentalCar_X['Parking']/rentalCar_X['Deplaned']
rentalCar_X['Ground'] = rentalCar_X['Ground'].transform(func='square')

rentalCar_X = rentalCar_X.transform(func='square')

# Storing only needed features
rentalCar_X = rentalCar_X[['Month','Transfer', 'Destination', 'Concession', 'Parking', 'Ground']].copy()
rentalCar_y = df.loc[:,['Rental Car']]

rentalCar_X.corr()
plt.figure(figsize=(14,10)),sns.heatmap(rentalCar_X.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)

# Split TRAIN and TEST data
rentalCar_X_train = rentalCar_X[:'Feb17']
rentalCar_X_test = rentalCar_X['Mar17':]
rentalCar_y_train = rentalCar_y[:'Feb17']
rentalCar_y_test = rentalCar_y['Mar17':]

# Check t-test
import scipy.stats as stats
stats.ttest_1samp(a = rentalCar_X_train, popmean = rentalCar_X.mean())

# Run Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(rentalCar_X_train,rentalCar_y_train)

# Predict
predict_rentalCar = lm.predict(rentalCar_X_test)

# Check R^2
lm.score(rentalCar_X,rentalCar_y)

# Check Intercept
lm.intercept_

# Check p-value again

stats.mstats.linregress(rentalCar_y_test,predict_rentalCar)

# ----------------------------------- Ground ----------------------------------

ground_X = df.loc[:,['Month', 'Enplaned', 'Deplaned', 'Transfer', 'Originating','Destination', 'Concession', 'Parking', 'Rental Car', 'Ground','Origin + Destin']]
ground_X.columns

# Heatmap to check correlation of X. Using this, we can decide which features to keep in training and testing dataset
plt.figure(figsize=(14,10)),sns.heatmap(ground_X.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)


# Transformations
ground_X = ground_X.transform(func='square')

# Storing only needed features
ground_X = ground_X[['Transfer', 'Originating', 'Parking', 'Ground']].copy()
ground_y = df.loc[:,['Ground']]

ground_X.corr()
plt.figure(figsize=(14,10)),sns.heatmap(ground_X.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)

# Split TRAIN and TEST data
ground_X_train = ground_X[:'Feb17']
ground_X_test = ground_X['Mar17':]
ground_y_train = ground_y[:'Feb17']
ground_y_test = ground_y['Mar17':]

# Check p-value
import scipy.stats as stats
stats.ttest_1samp(a = ground_X_train, popmean = ground_X.mean())

# Run Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(ground_X_train,ground_y_train)

# Predict
predict_ground = lm.predict(ground_X_test)

# Check R^2
lm.score(ground_X,ground_y)

# Check Intercept
lm.intercept_

# Check p-value again
stats.mstats.linregress(ground_y_test,predict_ground)


#------------------------------------------------------------------------------



        
