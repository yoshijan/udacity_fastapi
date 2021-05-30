from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import os
import pickle

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using
    precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def save_model(model, filename, basepath='starter/model'):
    """ save model as pickle file.

    Inputs
    ------
    model : ???
        Trained machine learning model or the encoder.
    filename : string
        filename which is used to save model .
    basepath: string
        folder for model storage using default one if not specified otherwise
    """
    path = os.path.join(basepath, filename)
    pickle.dump(model, open(path, 'wb'))

def load_model(filename, basepath='starter/model'):
    """ save model as pickle file.

    Inputs
    ------
    filename : string
        filename which is used to save model .
    basepath: string
        folder for model storage using default one if not specified otherwise
    -----
    Returns
        model : ???
        Trained machine learning model or the encoder
    """
    path  = os.path.join(basepath, filename)
    model = pickle.load(open(path, 'rb'))

    return model
