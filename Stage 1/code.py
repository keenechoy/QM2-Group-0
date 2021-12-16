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
import folium
import geopandas as gpd
import branca.colormap as cm

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
figure_width, figure_height = 10, 10
aahcratedata = np.genfromtxt(filename, delimiter=',')
x_values = aahcratedata[:, 2]
y_values = aahcratedata[:, 1]
aahcratedata[np.isnan(aahcratedata)] = 0
aahcratedata[np.isinf(aahcratedata)] = 0

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
y_lobf = [x_lobf[0]*gradient + intercept, x_lobf[1]*gradient + intercept]
plt.figure(figsize=(figure_width, figure_height))
plt.plot(x_values, y_values, 'b.', x_lobf, y_lobf, 'r--')
plt.title('Proportion of Asian Population and AAHC rate per Asian capita by US States')
plt.ylabel('Proportion of Asian Population')
plt.xlabel('AAHC rate per Asian capita (in %)')
plt.savefig(outputname)

# Regression for AAHC rate per asian capita and Total state population
filename = '/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/Output/aahcratetotalregress.csv'
outputname = '/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/Output/aahcratetotalregression.png'
figure_width, figure_height = 10, 10
aahcratedata = np.genfromtxt(filename, delimiter=',')
x_values = aahcratedata[:, 2]
y_values = aahcratedata[:, 1]
aahcratedata[np.isnan(aahcratedata)] = 0
aahcratedata[np.isinf(aahcratedata)] = 0

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
y_lobf = [x_lobf[0]*gradient + intercept, x_lobf[1]*gradient + intercept]
plt.figure(figsize=(figure_width, figure_height))
plt.plot(x_values, y_values, 'b.', x_lobf, y_lobf, 'r--')
plt.title('Total population and AAHC rate per Asian capita by US States')
plt.ylabel('Total population')
plt.xlabel('AAHC rate per Asian capita (in %)')
plt.savefig(outputname)

# Produce Stage 1 results choropleth map
map = gpd.read_file('/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/cb_2018_us_state_500k/cb_2018_us_state_500k.shp')
map = map[['GEOID', 'NAME', 'geometry']]
map.rename(columns={'NAME': 'State'}, inplace=True)

# Produce Asian Population layer on Stage 1 choropleth map
asianpop = race_demo.filter(['State', 'Asian alone'])
asianpop.rename(columns={'Asian alone': 'Asian Population'}, inplace=True)
mapasianpop = map.merge(asianpop, left_on='State', right_on='State', how='inner')

x_map=map.centroid.x.mean()
y_map=map.centroid.y.mean()
stage1map = folium.Map(location=[y_map, x_map], zoom_start=4,tiles=None)
folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(stage1map)
folium.Choropleth(
 geo_data=mapasianpop,
 name='Asian Population',
 data=mapasianpop,
 columns=['State','Asian Population'],
 key_on="feature.properties.State",
 fill_color='YlGnBu',
 bins=(0, 50000, 100000, 250000, 500000, 1000000, 6100000),
 fill_opacity=1,
 line_opacity=0.2,
 legend_name='Number of asian population',
 smooth_factor=0,
 overlay=True
).add_to(stage1map)

# Produce Asian Population Proportion layer on Stage 1 choropleth map
asianpopprop = race_demo.filter(['State', 'Proportion of Asian Population'])
mapasianpopprop = map.merge(asianpopprop, left_on='State', right_on='State', how='inner')

folium.Choropleth(
 geo_data=mapasianpopprop,
 name='Proportion of Asian Population',
 data=mapasianpopprop,
 columns=['State','Proportion of Asian Population'],
 key_on="feature.properties.State",
 fill_color='YlGnBu',
 bins=(0, 0.015, 0.03, 0.045, 0.06, 0.09, 0.38),
 fill_opacity=1,
 line_opacity=0.2,
 legend_name='Proportion of Asian Population',
 smooth_factor=0,
 overlay=True
).add_to(stage1map)

# Produce AAHC Cases layer on Stage 1 choropleth map
aahccases = aahc2020.filter(['State', 'AAHC number 2020'])
aahccases.drop_duplicates(ignore_index=True, inplace=True)
mapaahccases = map.merge(aahccases, left_on='State', right_on='State', how='inner')

folium.Choropleth(
 geo_data=mapaahccases,
 name='Number of AAHC cases in 2020',
 data=mapaahccases,
 columns=['State','AAHC number 2020'],
 key_on="feature.properties.State",
 fill_color='YlGnBu',
 bins=(0, 2, 5, 10, 20, 100),
 fill_opacity=1,
 line_opacity=0.2,
 legend_name='Number of AAHC cases in 2020',
 smooth_factor=0,
 overlay=True
).add_to(stage1map)

# Produce AAHC Rates layer on Stage 1 choropleth map
mapaahcrate = map.merge(aahcrate, left_on='State', right_on='State', how='inner')
mapaahcrate.drop(['Asian alone', 'Proportion of Asian Population', 'AAHC number 2020'], axis=1, inplace=True)

folium.Choropleth(
 geo_data=mapaahcrate,
 name='AAHC rate per Asian capita in %',
 data=mapaahcrate,
 columns=['State','AAHC rate per Asian capita in %'],
 key_on="feature.properties.State",
 fill_color='YlGnBu',
 bins=(0, 0.001, 0.002, 0.003, 0.005, 0.01, 0.025),
 fill_opacity=1,
 line_opacity=0.2,
 legend_name='AAHC rate per Asian capita in %',
 smooth_factor=0,
 overlay=True
).add_to(stage1map)

# Adding interactive layer on Stage 1 choropleth map
interactive = mapasianpop
interactive['Proportion of Asian Population'] = mapasianpopprop['Proportion of Asian Population']
interactive = interactive.merge(mapaahccases, left_on='geometry', right_on='geometry', how='left')
interactive.drop(['GEOID_y', 'State_y'], axis=1, inplace=True)
interactive = interactive.merge(mapaahcrate, left_on='geometry', right_on='geometry', how='left')
interactive.drop(['GEOID', 'State'], axis=1, inplace=True)
interactive.rename(columns={'GEOID_x': 'GEOID', 'State_x': 'State'}, inplace=True)
interactive = interactive.replace(np.NaN, 'No record')

style_function = lambda x: {'fillColor': '#ffffff',
                            'color':'#000000',
                            'fillOpacity': 0.1,
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000',
                                'color':'#000000',
                                'fillOpacity': 0.50,
                                'weight': 0.1}
interactivelayer = folium.features.GeoJson(
    interactive,
    style_function=style_function,
    control=False,
    highlight_function=highlight_function,
    tooltip=folium.features.GeoJsonTooltip(
        fields=['State', 'Asian Population', 'Proportion of Asian Population', 'AAHC number 2020', 'AAHC rate per Asian capita in %'],
        aliases=['State: ','Number of Asian population: ','Proportion of Asian population: ','Number of AAHC cases in 2020: ','AAHC rate per Asian capita in %: '],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
    ))
stage1map.add_child(interactivelayer)
stage1map.keep_in_front(interactivelayer)
folium.LayerControl(collapsed=False).add_to(stage1map)

# Save the Stage 1 Choropleth map as html file
stage1map.save('/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/Output/stage1map.html')

# Saving Asian population dataframe as csv for cartogram
asianpop.to_csv('/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/Stage 1/Output/asianpop.csv')