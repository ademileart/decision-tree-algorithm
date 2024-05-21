import pandas as pd
import enums as en
import plotly.express as px
from datetime import datetime
from decimal import Decimal
ts = 30883978148.882
timestamp = datetime.fromtimestamp(ts)
result = timestamp.second + Decimal(timestamp.microsecond)/1000000
print(result)
dataset = pd.read_csv('../healthcare-dataset-stroke-data.csv', sep=',', encoding='cp1252')

# remove missing values
dataset.dropna(inplace=True)

# change the age to numeric
dataset['age'] = pd.to_numeric(dataset.age, errors='coerce')

# row count
rowCount = dataset.shape[0]

# count males, females and not mentioned and draw pie chart
maleDataCount = dataset['gender'].eq(en.Gender.MALE.value).sum()
femaleDataCount = dataset['gender'].eq(en.Gender.FEMALE.value).sum()
notMentioned = dataset['gender'].eq(en.Gender.OTHER.value).sum()

fig = px.pie(dataset, values=[maleDataCount, femaleDataCount, notMentioned], names=['Male', 'Female', 'Not Mentioned'])
fig.show()
