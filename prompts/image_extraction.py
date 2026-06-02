import pandas as pd
import numpy as np
import requests

df = pd.read_csv('Obs_final.csv')

i=0
for i in range(0,50):
    url = df.iloc[i, 4]
    uuid = df.iloc[i, 0]
    response = requests.get(url)
    with open(f"data/image{uuid}.jpg", "wb") as f:
        f.write(response.content)
    f.close()