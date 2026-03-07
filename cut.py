import pandas as pd

df = pd.read_csv('data/dataCleanSCIE.csv')

df_cut = df.head(5)

df_cut.to_csv('data/dataCleanSCIE_Cut.csv', index=False)