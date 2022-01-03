# Importing packages
import pandas as pd
import numpy as np
import statsmodels.api as sms
import jenkspy
import shapely
import fiona
import pyproj
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import folium
import folium.plugins
import geopandas as gpd
import branca.colormap as cm

# STAGE1

# Setting dataframes
path1 = './Data/Stage 1/Input/us-state-pop-by-race.csv'
race_demo = pd.read_csv(path1, low_memory=False)

path2 = './Data/Stage 1/Input/hate_crime.csv'
crime_data = pd.read_csv(path2, low_memory=False)

# Creating a 2020 AAHC dataframe
aahc2020 = crime_data[crime_data['DATA_YEAR'] == 2020]
aahc2020 = aahc2020[aahc2020['BIAS_DESC'] == 'Anti-Asian']
aahc2020.reset_index(drop=True, inplace=True)

# Assigning 'State' attribute for 'Federal' cases by referring to the ORI
aahc2020.at[106:109, 'STATE_NAME'] = 'California'
aahc2020.at[110, 'STATE_NAME'] = 'Colorado'
aahc2020.at[111:112, 'STATE_NAME'] = 'Connecticut'
aahc2020.at[113:114, 'STATE_NAME'] = 'Montana'
aahc2020.at[115, 'STATE_NAME'] = 'New Mexico'
aahc2020.at[116, 'STATE_NAME'] = 'Texas'
aahc2020.at[117, 'STATE_NAME'] = 'Washington'

# Merging crime dataset with state population dataset
aahc2020.rename(columns={'STATE_NAME': 'State'}, inplace=True)
race_demo.rename(columns={'Label': 'State'}, inplace=True)
aahc2020 = aahc2020.merge(race_demo, left_on='State', right_on='State', how='left')

# Removing columns we don't need and cleaning data types
aahc2020.drop(['PUB_AGENCY_UNIT', 'Anti-Asian Hate Crime 2020', 'Native Hawaiian and Other Pacific Islander alone',
               'Asian and NHOPI', 'Proportion of NHOPI', 'Proportion of Asian and NHOPI'], axis=1, inplace=True)

# Counting number of AAHC by state
statecount = aahc2020.value_counts(subset='State').to_frame()
statecount.rename(columns={0: 'AAHC number 2020'}, inplace=True)
aahc2020 = aahc2020.merge(statecount, left_on='State', right_index=True, how='left')

# Calculating AAHC rates by state
aahc2020['AAHC rate per Asian capita in %'] = (aahc2020['AAHC number 2020']) / (aahc2020['Asian alone']) * 100
aahc2020['AAHC rate per capita in %'] = (aahc2020['AAHC number 2020']) / (aahc2020['Total']) * 100

# Creating dataframe for Stage 1 key data
stage1keydata = race_demo.filter(['State', 'Total', 'Asian alone', 'Proportion of Asian Population'])
stage1keydata.rename(columns={'Asian alone': 'Asian Population'}, inplace=True)
stage1keydata.drop_duplicates(ignore_index=True, inplace=True)
aahc2020_1 = aahc2020.filter(['State', 'AAHC number 2020', 'AAHC rate per Asian capita in %'])
aahc2020_1.drop_duplicates(ignore_index=True, inplace=True)
stage1keydata = stage1keydata.merge(aahc2020_1, left_on='State', right_on='State', how='left')

# Creating dataframes for regression and standardising data
stage1keydata_reg = aahc2020.filter(
    ['State', 'Total', 'Asian alone', 'Proportion of Asian Population', 'AAHC rate per Asian capita in %', 'AAHC number 2020'])
stage1keydata_reg.drop_duplicates(ignore_index=True, inplace=True)
stage1keydata_reg.rename(columns={'Asian alone': 'Asian Population'}, inplace=True)

regression1 = pd.DataFrame()
regression1['Asian Population (Standardised)'] = (stage1keydata_reg['Asian Population'] - stage1keydata_reg['Asian Population'].min()) / (stage1keydata_reg['Asian Population'].max() - stage1keydata_reg['Asian Population'].min())
regression1['Proportion of Asian Population (Standardised)'] = (stage1keydata_reg['Proportion of Asian Population'] - stage1keydata_reg['Proportion of Asian Population'].min()) / (stage1keydata_reg['Proportion of Asian Population'].max() - stage1keydata_reg['Proportion of Asian Population'].min())
regression1['Total (Standardised)'] = (stage1keydata_reg['Total'] - stage1keydata_reg['Total'].min()) / (stage1keydata_reg['Total'].max() - stage1keydata_reg['Total'].min())
regression1['AAHC rate per Asian capita in % (Standardised)'] = (stage1keydata_reg['AAHC rate per Asian capita in %'] - stage1keydata_reg['AAHC rate per Asian capita in %'].min()) / (stage1keydata_reg['AAHC rate per Asian capita in %'].max() - stage1keydata_reg['AAHC rate per Asian capita in %'].min())
regression1.to_csv(path_or_buf='./Data/Stage 1/Output/regression1.csv')

# Regression for AAHC rate per asian capita and Asian population
inputname = './Data/Stage 1/Output/regression1.csv'
outputname = './Data/Stage 1/Output/aahcrateasianregression.png'
figure_width, figure_height = 10, 10
regressiondata = np.genfromtxt(inputname, delimiter=',')
x_values = regressiondata[1:,1]
y_values = regressiondata[1:,4]
regressiondata[np.isnan(regressiondata)] = 0
regressiondata[np.isinf(regressiondata)] = 0

X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
print()
print('AAHC rate and Asian population regression summary')
print(regression_model_b.summary())
print()

gradient = regression_model_b.params[1]
intercept = regression_model_b.params[0]
Rsquared = regression_model_b.rsquared
MSE = regression_model_b.mse_resid
pvalue = regression_model_b.f_pvalue
print("gradient  =", regression_model_b.params[1])
print("intercept =", regression_model_b.params[0])
print("Rsquared  =", regression_model_b.rsquared)
print("MSE       =", regression_model_b.mse_resid)
print("pvalue    =", regression_model_b.f_pvalue)

x_lobf = [min(x_values), max(x_values)]
y_lobf = [x_lobf[0] * gradient + intercept, x_lobf[1] * gradient + intercept]
plt.figure(figsize=(figure_width, figure_height))
plt.plot(x_values, y_values, 'b.', x_lobf, y_lobf, 'r--')
plt.title('Asian Population and AAHC rate per Asian capita by US States')
plt.xlabel('Asian Population (Standardised)')
plt.ylabel('AAHC rate per Asian capita (Standardised)')
plt.savefig(outputname)

# Regression for AAHC rate per asian capita and Asian population proportion
inputname = './Data/Stage 1/Output/regression1.csv'
outputname = './Data/Stage 1/Output/aahcrateproportionregression.png'
figure_width, figure_height = 10, 10
regressiondata = np.genfromtxt(inputname, delimiter=',')
x_values = regressiondata[1:,2]
y_values = regressiondata[1:,4]
regressiondata[np.isnan(regressiondata)] = 0
regressiondata[np.isinf(regressiondata)] = 0

X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
print()
print('AAHC rate and Asian population proportion regression summary')
print(regression_model_b.summary())
print()

gradient = regression_model_b.params[1]
intercept = regression_model_b.params[0]
Rsquared = regression_model_b.rsquared
MSE = regression_model_b.mse_resid
pvalue = regression_model_b.f_pvalue
print("gradient  =", regression_model_b.params[1])
print("intercept =", regression_model_b.params[0])
print("Rsquared  =", regression_model_b.rsquared)
print("MSE       =", regression_model_b.mse_resid)
print("pvalue    =", regression_model_b.f_pvalue)

x_lobf = [min(x_values), max(x_values)]
y_lobf = [x_lobf[0] * gradient + intercept, x_lobf[1] * gradient + intercept]
plt.figure(figsize=(figure_width, figure_height))
plt.plot(x_values, y_values, 'b.', x_lobf, y_lobf, 'r--')
plt.title('Proportion of Asian Population and AAHC rate per Asian capita by US States')
plt.xlabel('Proportion of Asian Population (Standardised')
plt.ylabel('AAHC rate per Asian capita (Standardised)')
plt.savefig(outputname)

# Regression for AAHC rate per asian capita and Total state population
inputname = './Data/Stage 1/Output/regression1.csv'
outputname = './Data/Stage 1/Output/aahcratetotalregression.png'
figure_width, figure_height = 10, 10
regressiondata = np.genfromtxt(inputname, delimiter=',')
x_values = regressiondata[1:,3]
y_values = regressiondata[1:,4]
regressiondata[np.isnan(regressiondata)] = 0
regressiondata[np.isinf(regressiondata)] = 0

X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
print()
print('AAHC rate and total population regression summary')
print(regression_model_b.summary())
print()

gradient = regression_model_b.params[1]
intercept = regression_model_b.params[0]
Rsquared = regression_model_b.rsquared
MSE = regression_model_b.mse_resid
pvalue = regression_model_b.f_pvalue
print("gradient  =", regression_model_b.params[1])
print("intercept =", regression_model_b.params[0])
print("Rsquared  =", regression_model_b.rsquared)
print("MSE       =", regression_model_b.mse_resid)
print("pvalue    =", regression_model_b.f_pvalue)

x_lobf = [min(x_values), max(x_values)]
y_lobf = [x_lobf[0] * gradient + intercept, x_lobf[1] * gradient + intercept]
plt.figure(figsize=(figure_width, figure_height))
plt.plot(x_values, y_values, 'b.', x_lobf, y_lobf, 'r--')
plt.title('Total population and AAHC rate per Asian capita by US States')
plt.xlabel('Total population (Standardised')
plt.ylabel('AAHC rate per Asian capita (Standardised)')
plt.savefig(outputname)

# Produce Stage 1 results choropleth map
map = gpd.read_file('./Data/Stage 1/Input/cb_2018_us_state_500k/cb_2018_us_state_500k.shp')
map = map[['GEOID', 'NAME', 'geometry']]
map.rename(columns={'NAME': 'State'}, inplace=True)
map.drop([13,37,38,44,45], axis=0, inplace=True)

# Produce Asian Population layer on Stage 1 choropleth map
mapstage1keydata = map.merge(stage1keydata, left_on='State', right_on='State', how='left')

x_map = map.centroid.x.mean()
y_map = map.centroid.y.mean()
stage1map = folium.Map(location=[y_map, x_map], zoom_start=4, tiles=None)
folium.TileLayer('CartoDB positron', name="Light Map", control=False).add_to(stage1map)

asianpopmap = folium.features.Choropleth(
    geo_data=mapstage1keydata,
    name='Asian Population',
    data=mapstage1keydata,
    columns=['State', 'Asian Population'],
    key_on="feature.properties.State",
    fill_color='Blues',
    bins=pd.Series(jenkspy.jenks_breaks(mapstage1keydata['Asian Population'], nb_class=7)),
    fill_opacity=0.8,
    line_opacity=0.2,
    legend_name='Number of asian population',
    smooth_factor=0,
    overlay=True,
    show=True
)
stage1map.add_child(asianpopmap)

# Produce Asian Population Proportion layer on Stage 1 choropleth map
asianpoppropmap = folium.features.Choropleth(
    geo_data=mapstage1keydata,
    name='Proportion of Asian Population',
    data=mapstage1keydata,
    columns=['State', 'Proportion of Asian Population'],
    key_on="feature.properties.State",
    fill_color='Blues',
    bins=(0, 0.015, 0.03, 0.045, 0.06, 0.09, 0.38),
    fill_opacity=0.8,
    line_opacity=0.2,
    legend_name='Proportion of Asian Population',
    smooth_factor=0,
    overlay=True,
    show=False
)
stage1map.add_child(asianpoppropmap)

# Produce AAHC Cases layer on Stage 1 choropleth map
aahccasesmap = folium.features.Choropleth(
    geo_data=mapstage1keydata,
    name='Number of AAHC cases in 2020',
    data=mapstage1keydata,
    columns=['State', 'AAHC number 2020'],
    key_on="feature.properties.State",
    fill_color='Blues',
    bins=(0, 2, 5, 10, 20, 100),
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Number of AAHC cases in 2020',
    smooth_factor=0,
    overlay=True,
    show=False
)
stage1map.add_child(aahccasesmap)

# Produce AAHC Rates layer on Stage 1 choropleth map
aahcratemap = folium.features.Choropleth(
    geo_data=mapstage1keydata,
    name='AAHC rate per Asian capita in %',
    data=mapstage1keydata,
    columns=['State', 'AAHC rate per Asian capita in %'],
    key_on="feature.properties.State",
    fill_color='Blues',
    bins=(0, 0.001, 0.002, 0.003, 0.005, 0.01, 0.025),
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='AAHC rate per Asian capita in %',
    smooth_factor=0,
    overlay=True,
    show=False
)
stage1map.add_child(aahcratemap)

# Adding interactive layer on Stage 1 choropleth map
mapstage1keydata = mapstage1keydata.replace(np.NaN, 'No record')
style_function = lambda x: {'fillColor': '#ffffff',
                            'color': '#000000',
                            'fillOpacity': 0.1,
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000',
                                'color': '#000000',
                                'fillOpacity': 0.50,
                                'weight': 0.1}
interactivemap = folium.features.GeoJson(
    mapstage1keydata,
    style_function=style_function,
    control=False,
    overlay=True,
    show=True,
    highlight_function=highlight_function,
    tooltip=folium.features.GeoJsonTooltip(
        fields=['State', 'Asian Population', 'Proportion of Asian Population', 'AAHC number 2020',
                'AAHC rate per Asian capita in %'],
        aliases=['State: ', 'Asian population: ', 'Proportion of Asian population: ', 'Number of AAHC cases in 2020: ',
                 'AAHC rate per Asian capita in %: '],
        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",

    ))
stage1map.add_child(interactivemap)
stage1map.keep_in_front(interactivemap)
folium.LayerControl(collapsed=False, autoZIndex=False).add_to(stage1map)

# Save the Stage 1 Choropleth map as html file
stage1map.save('./Data/Stage 1/Output/stage1map.html')

# Saving Stage 1 key data dataframe as csv
stage1keydata.to_csv('./Data/Stage 1/Output/stage1keydata.csv')

# STAGE2

# Setting dataframes from imported data
path = './Data/Stage 2/Input/Popular vote backend - Sheet1.csv'
popularvote = pd.read_csv(path, low_memory=False)

path = './Data/Stage 2/Input/School Dropout data.csv'
dropout = pd.read_csv(path, low_memory=False)

path = './Data/Stage 2/Input/US states- income QM database.xlsx - Sheet1.csv'
income = pd.read_csv(path, low_memory=False)

# Cleaning dataframes
dropout = dropout.filter(['State', 'Total'])
popularvote = popularvote.filter(['state', 'called', 'dem_this_margin'])
popularvote['dem_this_margin'] = popularvote['dem_this_margin'].str.rstrip('%').astype('float')

# Creating Stage 2 key data dataframe
stage2keydata = dropout.merge(popularvote, left_on='State', right_on='state', how='inner')
stage2keydata = stage2keydata.merge(income, left_on='State', right_on='State', how='inner')
stage2keydata.drop(['called'], axis=1, inplace=True)

# Creating side-by-side choropleth maps showing datasets
x_map = map.centroid.x.mean()
y_map = map.centroid.y.mean()
stage2map = folium.plugins.DualMap(zoom_start=4, tiles=False, layout='vertical')
folium.TileLayer('CartoDB positron', name="Light Map", control=False).add_to(stage2map.m1)
folium.TileLayer('CartoDB positron', name="Light Map", control=False).add_to(stage2map.m2)

# Copying the Stage 1 maps to the left side
asianpopmap.add_to(stage2map.m1)
asianpoppropmap.add_to(stage2map.m1)
aahccasesmap.add_to(stage2map.m1)
aahcratemap.add_to(stage2map.m1)
interactivemap.add_to(stage2map.m1)

# Adding Stage 2 maps to the right side
mapstage2keydata = map.merge(stage2keydata, left_on='State', right_on='State', how='inner')
folium.features.Choropleth(
    geo_data=mapstage2keydata,
    name='Democratic margin in 2020 Presidential Election',
    data=mapstage2keydata,
    columns=['State', 'dem_this_margin'],
    key_on="feature.properties.State",
    fill_color='RdBu',
    bins=(-50, -25, -15, -10, -5, 0, 5, 10, 15, 25, 90),
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Democratic margin in 2020 Presidential Election',
    smooth_factor=0,
    overlay=True,
    show=True
).add_to(stage2map.m2)

folium.features.Choropleth(
    geo_data=mapstage2keydata,
    name='Median Annual Income (USD)',
    data=mapstage2keydata,
    columns=['State', 'Median annual income (USD)'],
    key_on="feature.properties.State",
    fill_color='Blues',
    bins=(24000, 29000, 34000, 39000, 44000, 57000),
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Median Annual Income (USD)',
    smooth_factor=0,
    overlay=True,
    show=False
).add_to(stage2map.m2)

folium.features.Choropleth(
    geo_data=mapstage2keydata,
    name='High School Dropout Rates (%)',
    data=mapstage2keydata,
    columns=['State', 'Total'],
    key_on="feature.properties.State",
    fill_color='Blues',
    bins=(3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 9.0),
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='High School Dropout Rates (%)',
    smooth_factor=0,
    overlay=True,
    show=False
).add_to(stage2map.m2)

# Adding interactive layer to the right side
interactivemap2 = folium.features.GeoJson(
    mapstage2keydata,
    style_function=style_function,
    control=False,
    overlay=True,
    show=True,
    highlight_function=highlight_function,
    tooltip=folium.features.GeoJsonTooltip(
        fields=['State', 'dem_this_margin', 'Median annual income (USD)', 'Total'],
        aliases=['State: ', 'Democratic margin in 2020 Presidential Election: ', 'Median Annual Income (USD): ',
                 'High School Dropout Rates (%): '],
        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",

    ))
interactivemap2.add_to(stage2map.m2)
stage2map.m1.keep_in_front(interactivemap)
stage2map.m2.keep_in_front(interactivemap2)

# Adding layer control to stage 2 map
folium.LayerControl(collapsed=False, autoZIndex=False).add_to(stage2map.m1)
folium.LayerControl(collapsed=False, autoZIndex=False).add_to(stage2map.m2)

# Exporting stage 2 map
stage2map.save('./Data/Stage 2/Output/stage2map.html')

# Regression for Income and AAHC rate
stage2keydata_reg = stage1keydata_reg.filter(['State', 'AAHC rate per Asian capita in %'])
stage2keydata_reg = stage2keydata_reg.merge(stage2keydata, left_on='State', right_on='State', how='inner')
regression2 = pd.DataFrame()
regression2['Median annual income (Standardised)'] = (stage2keydata_reg['Median annual income (USD)'] - stage2keydata_reg['Median annual income (USD)'].mean())/(stage2keydata_reg['Median annual income (USD)'].std())
regression2['dem_this_margin (Standardised)'] = (stage2keydata_reg['dem_this_margin'] - stage2keydata_reg['dem_this_margin'].mean())/(stage2keydata_reg['dem_this_margin'].std())
regression2['Dropout rates (Standardised)'] = (stage2keydata_reg['Total'] - stage2keydata_reg['Total'].mean())/(stage2keydata_reg['Total'].std())
regression2['AAHC rate per Asian capita in % (Standardised)'] = (stage1keydata_reg['AAHC rate per Asian capita in %'] - stage1keydata_reg['AAHC rate per Asian capita in %'].mean())/(stage1keydata_reg['AAHC rate per Asian capita in %'].std())
regression2.to_csv(path_or_buf='./Data/Stage 2/Output/regression2.csv')

filename = './Data/Stage 2/Output/regression2.csv'
outputname = './Data/Stage 2/Output/incomeregression.png'
figure_width, figure_height = 10, 10
regressiondata = np.genfromtxt(filename, delimiter=',')
x_values = regressiondata[1:, 1]
y_values = regressiondata[1:, 4]
regressiondata[np.isnan(regressiondata)] = 0
regressiondata[np.isinf(regressiondata)] = 0

X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
print()
print('AAHC rate and income regression summary')
print(regression_model_b.summary())
print()

gradient = regression_model_b.params[1]
intercept = regression_model_b.params[0]
Rsquared = regression_model_b.rsquared
MSE = regression_model_b.mse_resid
pvalue = regression_model_b.f_pvalue
print("gradient  =", regression_model_b.params[1])
print("intercept =", regression_model_b.params[0])
print("Rsquared  =", regression_model_b.rsquared)
print("MSE       =", regression_model_b.mse_resid)
print("pvalue    =", regression_model_b.f_pvalue)

x_lobf = [min(x_values), max(x_values)]
y_lobf = [x_lobf[0] * gradient + intercept, x_lobf[1] * gradient + intercept]
plt.figure(figsize=(figure_width, figure_height))
plt.plot(x_values, y_values, 'b.', x_lobf, y_lobf, 'r--')
plt.title('Median annual income and AAHC rate per Asian capita by US States')
plt.xlabel('Median annual income (Standardised)')
plt.ylabel('AAHC rate per Asian capita (Standardised)')
plt.savefig(outputname)

# Regression for Dropout rate and AAHC rate
filename = './Data/Stage 2/Output/regression2.csv'
outputname = './Data/Stage 2/Output/dropoutregression.png'
figure_width, figure_height = 10, 10
regressiondata = np.genfromtxt(filename, delimiter=',')
x_values = regressiondata[1:, 3]
y_values = regressiondata[1:, 4]
regressiondata[np.isnan(regressiondata)] = 0
regressiondata[np.isinf(regressiondata)] = 0

X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
print()
print('AAHC rate and dropout regression summary')
print(regression_model_b.summary())
print()

gradient = regression_model_b.params[1]
intercept = regression_model_b.params[0]
Rsquared = regression_model_b.rsquared
MSE = regression_model_b.mse_resid
pvalue = regression_model_b.f_pvalue
print("gradient  =", regression_model_b.params[1])
print("intercept =", regression_model_b.params[0])
print("Rsquared  =", regression_model_b.rsquared)
print("MSE       =", regression_model_b.mse_resid)
print("pvalue    =", regression_model_b.f_pvalue)

x_lobf = [min(x_values), max(x_values)]
y_lobf = [x_lobf[0] * gradient + intercept, x_lobf[1] * gradient + intercept]
plt.figure(figsize=(figure_width, figure_height))
plt.plot(x_values, y_values, 'b.', x_lobf, y_lobf, 'r--')
plt.title('High School Dropout Rates and AAHC rate per Asian capita by US States')
plt.xlabel('High School Dropout Rates (Standardised)')
plt.ylabel('AAHC rate per Asian capita (Standardised)')
plt.savefig(outputname)

# Regression for popular vote and AAHC rate
filename = './Data/Stage 2/Output/regression2.csv'
outputname = './Data/Stage 2/Output/popularvoteregression.png'
figure_width, figure_height = 10, 10
regressiondata = np.genfromtxt(filename, delimiter=',')
x_values = regressiondata[1:, 2]
y_values = regressiondata[1:, 4]
regressiondata[np.isnan(regressiondata)] = 0
regressiondata[np.isinf(regressiondata)] = 0

X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
print()
print('AAHC rate and popular vote regression summary')
print(regression_model_b.summary())
print()

gradient = regression_model_b.params[1]
intercept = regression_model_b.params[0]
Rsquared = regression_model_b.rsquared
MSE = regression_model_b.mse_resid
pvalue = regression_model_b.f_pvalue
print("gradient  =", regression_model_b.params[1])
print("intercept =", regression_model_b.params[0])
print("Rsquared  =", regression_model_b.rsquared)
print("MSE       =", regression_model_b.mse_resid)
print("pvalue    =", regression_model_b.f_pvalue)

x_lobf = [min(x_values), max(x_values)]
y_lobf = [x_lobf[0] * gradient + intercept, x_lobf[1] * gradient + intercept]
plt.figure(figsize=(figure_width, figure_height))
plt.plot(x_values, y_values, 'b.', x_lobf, y_lobf, 'r--')
plt.title('Democratic margin in 2020 Presidential Election and AAHC rate per Asian capita by US States')
plt.xlabel('Democratic margin in 2020 Presidential Election (Standardised)')
plt.ylabel('AAHC rate per Asian capita (Standardised)')
plt.savefig(outputname)

# Saving Stage 2 key data dataframe as csv
stage2keydata.to_csv('./Data/Stage 2/Output/stage2keydata.csv')