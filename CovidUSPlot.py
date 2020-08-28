import urllib.request
import plotly.express as px
import pandas as pd
import numpy as np
import json

with urllib.request.urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv', dtype={"fips": str, "state": str, "county": str})
df.loc[df['county'] == "New York City",'fips'] = "36061"
df = df.groupby(['fips', 'state', 'county']).sum() #fips are index of counties 
df = df.reset_index()
df['Deaths (log10)'] = np.log10(df['deaths'])

fig = px.choropleth(df, locations='Counties',
                        color='Deaths (log10)',
                        scope='usa',
                        geojson=counties,
                        hover_data=['deaths'])
fig.show()