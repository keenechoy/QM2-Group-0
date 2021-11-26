import pandas as pd

path = '/Users/keenechoy/PycharmProjects/QM2-Group-0/Data/us-state-pop-by-race.csv'
data = pd.read_csv(path, encoding='latin1')

print(data.head())
