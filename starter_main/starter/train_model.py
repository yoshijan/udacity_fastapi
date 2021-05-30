# Script to train machine learning model.

import numpy as np
import pandas as pd
# Add the necessary imports for the starter code.
import starter_main.starter.ml.data as mldata
import starter_main.starter.ml.model as mlmodel
from sklearn.model_selection import train_test_split


def train_model():
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
    print("start process test datafunction")
    X_test, y_test, encoder, lb = mldata.process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    # Train and save a model.
    model = mlmodel.train_model(X_train, y_train)
    preds_test = mlmodel.inference(model,X_test)
    #calc score
    precision, recall, fbeta =mlmodel.compute_model_metrics(y_test,preds_test)
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("fbeta: " + str(fbeta))

    print("store model")
    mlmodel.save_model(model, 'model.pkl')
    print("store encoder")
    mlmodel.save_model(encoder, 'encoder.pkl')
    print("store lb")
    mlmodel.save_model(lb, 'lb.pkl')

    #check to load model
    print("load model")
    loaded_model = mlmodel.load_model('model.pkl')

    print("load encoder")
    encoder = mlmodel.load_model('encoder.pkl')

    print("load lb")
    lb = mlmodel.load_model('lb.pkl')

    print(loaded_model.coef_.shape)

#seperate function for model performacne output on a set of categorical features
def silice_test_performance(cat_features):
    print("read pandas")
    data = pd.read_csv('starter_main/data/census_clean.csv')
    print("df created: " + str(data.shape))

    #add target variable

    cat_features_list = np.append(cat_features, "salary")

    #filter data
    data = data[cat_features_list]

    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = mldata.process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    print("start process test datafunction")
    X_test, y_test, encoder, lb = mldata.process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    # Train and save a model.
    model = mlmodel.train_model(X_train, y_train)
    preds_test = mlmodel.inference(model,X_test)
    #calc score
    precision, recall, fbeta =mlmodel.compute_model_metrics(y_test,preds_test)
    return precision, recall, fbeta