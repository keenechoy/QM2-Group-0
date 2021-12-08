# Importing packages
import pandas as pd
import numpy
import matplotlib as plt
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
aahc2020.drop(['PUB_AGENCY_UNIT', 'Anti-Asian Hate Crime 2020', 'Native Hawaiian and Other Pacific Islander alone', 'Asian and NHOPI', 'Proportion of NHOPI', 'Proportion of Asian and NHOPI'], axis=1, inplace=True)

# Counting number of AAHC by state
statecount = aahc2020.value_counts(subset='State')
statecount = statecount.to_frame()
statecount.rename(columns={0: 'AAHC number 2020'}, inplace=True)
aahc2020 = aahc2020.merge(statecount, left_on='State', right_index=True, how='left')

# Counting AAHC rate per Asian capita of each state
aahc2020['AAHC rate per Asian capita in %'] = (aahc2020['AAHC number 2020'])/(aahc2020['Asian alone'])*100
aahcrate = aahc2020.filter(['State', 'AAHC rate per Asian capita in %'])
aahcrate.drop_duplicates(ignore_index=True, inplace=True)

# Produce Asian population graphics
import folium
asianpopmap = folium.Map(location=[40,-95],zoom_start=4)
asianpopmap.save('asianpopmap.html')
asianpop = aahc2020.filter(['State', 'Asian alone'])
asianpop.drop_duplicates(ignore_index=True, inplace=True)
asianpop.rename(columns={'Asian alone': 'Asian Population'}, inplace=True)

