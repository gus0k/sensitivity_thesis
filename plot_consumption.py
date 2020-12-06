#%%

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

df = pd.read_csv('home_data_2012-13.csv')
df['date'] = pd.to_datetime(df.date)

sum_ = df.groupby(df.date.dt.floor('d')).sum()


#%%
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=pd.melt(sum_), x='variable', y='value',
        order=sum_.mean().sort_values().index, ax=ax)
ax.set_xticks(ax.get_xticks()[::10])
ax.set_xlabel('Customer ID')
ax.set_ylabel('Energy consumed in a day [kWh]')

matplotlib.use('pgf')
fig.savefig('consumption_players.pgf')


