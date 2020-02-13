import pandas as pd
import re

df = pd.read_csv('amlo.csv')

text = df.iloc[:,2]

entry = text[2].split('|')

for p in entry:

    # print(text[i] + "\n")

    m = re.search('<p class="p1">(.+?)</p>', p)

    if m:
        print(m.group(1) + "\n")
