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

# Assigning 'State' attribute for 'Federal' cases by looking at the ORI
if aahc2020['ORI'] == 'CAFBILA00' or aahc2020['ORI'] == 'CAFBISC00':
    aahc2020['STATE_NAME'] = 'California'

# Merging with state dataset
aahc2020.rename(columns={'STATE_NAME': 'State'}, inplace=True)
race_demo.rename(columns={'Label': 'State'}, inplace=True)
aahc2020 = aahc2020.merge(race_demo, left_on='State', right_on='State', how='left')

# Removing collumns we don't need and cleaning data types
aahc2020.drop(['PUB_AGENCY_UNIT', 'Anti-Asian Hate Crime 2020'], axis=1, inplace=True)

#Counting number of AAHC by state
statecount = aahc2020.value_counts(subset='State')
statecount = statecount.to_frame()


