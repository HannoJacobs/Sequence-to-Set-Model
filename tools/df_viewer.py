"""view STMaster"""

import pandas as pd
from display_df import display_df

df = pd.read_csv("Datasets/_dataset.csv")
display_df(df)
