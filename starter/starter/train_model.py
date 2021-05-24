# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import ml.data as mldata
import pandas as pd

# Add code to load in the data.
print("read pandas")
data = pd.read_csv('starter/data/census_clean.csv')
print("df created: " + str(data.shape))

# Optional enhancement,
# use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = mldata.process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
print("start process function")
# Train and save a model.
