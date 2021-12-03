# Importing packages
import pandas as pd
import matplotlib as plt
import plotly.express as px
import seaborn as sns

# Setting dataframes
path1 = './Data/Stage 1/us-state-pop-by-race.csv'
race_demo = pd.read_csv(path1, encoding='latin1')

path2 = './Data/Stage 1/hate_crime.csv'
crime_data= pd.read_csv(path2, encoding='latin1')

# Cleaning crime_data to only contain what we need
crime_data.head()


