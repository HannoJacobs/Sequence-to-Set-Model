"""view STMaster"""

import pandas as pd
from display_df import display_df

df = pd.read_csv("Datasets/dataset.csv")
display_df(df)
