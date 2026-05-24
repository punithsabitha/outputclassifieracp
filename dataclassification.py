# import libraries
import pandas as pd
from matplotlib import pyplot

# load dataset
df = pd.read_csv("Titanic.csv")

# remove null values
df = df.dropna()

# print dataset shape
print(df.shape)

# survived and dead people
count = df['Survived'].value_counts()
print(count)

# plot survived people
pyplot.scatter(df.index, df['Survived'])

# title and labels
pyplot.title("Survived Classification")
pyplot.xlabel("People")
pyplot.ylabel("Survived")

# show graph
pyplot.show()