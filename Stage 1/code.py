# Importing packages
import pandas as pd
import numpy as np
import statsmodels.api as sms

import shapely
# import fiona
import pyproj
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Setting dataframes
path1 = '/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/us-state-pop-by-race.csv'
race_demo = pd.read_csv(path1, low_memory=False)

path2 = '/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/hate_crime.csv'
crime_data = pd.read_csv(path2, low_memory=False)

# Creating a 2020 AAHC dataframe
aahc2020 = crime_data[crime_data['DATA_YEAR'] == 2020]
aahc2020 = aahc2020[aahc2020['BIAS_DESC'] == 'Anti-Asian']
aahc2020.reset_index(drop=True, inplace=True)

# Directly assigning 'State' attribute for 'Federal' cases by referring at the ORI
aahc2020.at[106, 'STATE_NAME'] = 'California'
aahc2020.at[107, 'STATE_NAME'] = 'California'
aahc2020.at[108, 'STATE_NAME'] = 'California'
aahc2020.at[109, 'STATE_NAME'] = 'California'
aahc2020.at[110, 'STATE_NAME'] = 'Colorado'
aahc2020.at[111, 'STATE_NAME'] = 'Connecticut'
aahc2020.at[112, 'STATE_NAME'] = 'Connecticut'
aahc2020.at[113, 'STATE_NAME'] = 'Montana'
aahc2020.at[114, 'STATE_NAME'] = 'Montana'
aahc2020.at[115, 'STATE_NAME'] = 'New Mexico'
aahc2020.at[116, 'STATE_NAME'] = 'Texas'
aahc2020.at[117, 'STATE_NAME'] = 'Washington'

# Merging with state dataset
aahc2020.rename(columns={'STATE_NAME': 'State'}, inplace=True)
race_demo.rename(columns={'Label': 'State'}, inplace=True)
aahc2020 = aahc2020.merge(race_demo, left_on='State', right_on='State', how='left')

# Removing collumns we don't need and cleaning data types
aahc2020.drop(['PUB_AGENCY_UNIT', 'Anti-Asian Hate Crime 2020', 'Native Hawaiian and Other Pacific Islander alone',
               'Asian and NHOPI', 'Proportion of NHOPI', 'Proportion of Asian and NHOPI'], axis=1, inplace=True)

# Counting number of AAHC by state
statecount = aahc2020.value_counts(subset='State')
statecount = statecount.to_frame()
statecount.rename(columns={0: 'AAHC number 2020'}, inplace=True)
aahc2020 = aahc2020.merge(statecount, left_on='State', right_index=True, how='left')

# Counting AAHC rate per Asian capita of each state
aahc2020['AAHC rate per Asian capita in %'] = (aahc2020['AAHC number 2020']) / (aahc2020['Asian alone']) * 100
aahcrate = aahc2020.filter(['State', 'Asian alone', 'Proportion of Asian Population', 'AAHC number 2020', 'AAHC rate per Asian capita in %'])
aahcrate.drop_duplicates(ignore_index=True, inplace=True)
aahcrateregress = aahcrate.filter(['Proportion of Asian Population', 'AAHC rate per Asian capita in %'])
aahcrateregress.to_csv(path_or_buf='/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/Output/aahcrateregress.csv')

# Counting AAHC rate per capita of each state
aahc2020['AAHC rate per capita in %'] = (aahc2020['AAHC number 2020']) / (aahc2020['Total']) * 100
aahcratetotal = aahc2020.filter(['State', 'Total', 'AAHC number 2020', 'AAHC rate per capita in %'])
aahcratetotal.drop_duplicates(ignore_index=True, inplace=True)
aahcratetotalregress = aahcratetotal.filter(['Total', 'AAHC rate per capita in %'])
aahcratetotalregress.to_csv(path_or_buf='/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/Output/aahcratetotalregress.csv')

# Regression for AAHC rate per asian capita and Asian population proportion
filename = '/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/Output/aahcrateregress.csv'
outputname = '/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/Output/aahcrateregression.png'
figure_width, figure_height = 10,10
aahcratedata = np.genfromtxt(filename, delimiter = ',')
x_values = aahcratedata[:,2]
y_values = aahcratedata[:,1]
aahcratedata[np.isnan(aahcratedata)] = 0
aahcratedata[np.isinf(aahcratedata)] = 0

X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
print()
print('AAHC rate and Asian population proportion regression summary')
print(regression_model_b.summary())
print()

gradient  = regression_model_b.params[1]
intercept = regression_model_b.params[0]
Rsquared  = regression_model_b.rsquared
MSE       = regression_model_b.mse_resid
pvalue    = regression_model_b.f_pvalue
print("gradient  =", regression_model_b.params[1])
print("intercept =", regression_model_b.params[0])
print("Rsquared  =", regression_model_b.rsquared)
print("MSE       =", regression_model_b.mse_resid)
print("pvalue    =", regression_model_b.f_pvalue)


x_lobf = [min(x_values),max(x_values)]
y_lobf = [x_lobf[0]*gradient + intercept,x_lobf[1]*gradient + intercept]
plt.figure(figsize=(figure_width,figure_height))
plt.plot(x_values,y_values,'b.',x_lobf,y_lobf,'r--')
plt.title('Proportion of Asian Population and AAHC rate per Asian capita by US States')
plt.ylabel('Proportion of Asian Population')
plt.xlabel('AAHC rate per Asian capita (in %)')
plt.savefig(outputname)

# Regression for AAHC rate per asian capita and Total state population

filename = '/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/Output/aahcratetotalregress.csv'
outputname = '/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/Output/aahcratetotalregression.png'
figure_width, figure_height = 10,10
aahcratedata = np.genfromtxt(filename, delimiter = ',')
x_values = aahcratedata[:,2]
y_values = aahcratedata[:,1]
aahcratedata[np.isnan(aahcratedata)] = 0
aahcratedata[np.isinf(aahcratedata)] = 0

X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
print()
print('AAHC rate and total population regression summary')
print(regression_model_b.summary())
print()

gradient  = regression_model_b.params[1]
intercept = regression_model_b.params[0]
Rsquared  = regression_model_b.rsquared
MSE       = regression_model_b.mse_resid
pvalue    = regression_model_b.f_pvalue
print("gradient  =", regression_model_b.params[1])
print("intercept =", regression_model_b.params[0])
print("Rsquared  =", regression_model_b.rsquared)
print("MSE       =", regression_model_b.mse_resid)
print("pvalue    =", regression_model_b.f_pvalue)


x_lobf = [min(x_values),max(x_values)]
y_lobf = [x_lobf[0]*gradient + intercept,x_lobf[1]*gradient + intercept]
plt.figure(figsize=(figure_width,figure_height))
plt.plot(x_values,y_values,'b.',x_lobf,y_lobf,'r--')
plt.title('Total population and AAHC rate per Asian capita by US States')
plt.ylabel('Total population')
plt.xlabel('AAHC rate per Asian capita (in %)')
plt.savefig(outputname)


#Produce Asian population graphics
import folium
import geopandas as gpd
import branca.colormap as cm

#asianpop = aahc2020.filter(['State', 'Asian alone'])
#asianpop.drop_duplicates(ignore_index=True, inplace=True)
#asianpop.rename(columns={'Asian alone': 'Asian Population'}, inplace=True)

#uno = gpd.read_file('/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stag3e 1/cb_2018_us_state_500k.shp')
#uno.head()