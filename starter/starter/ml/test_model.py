from sklearn import datasets
from starter.starter.ml.model import train_model
from starter.starter.ml.model import compute_model_metrics
from starter.starter.ml.model import inference
#
def test_train_model():
    #setup
    digits = datasets.load_digits()
    model = train_model(digits.data, digits.target)
    #test for type of the model
    model_name = type(model).__name__
    assert (model_name == 'LogisticRegression')

def test_compute_model_metrics():
    #setup
    y  = [0, 0, 0, 0, 0, 0]
    preds =  [0, 0, 0, 0, 0, 0]
    #assert for metric outputs
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision==1.0
    assert recall==1.0
    assert fbeta==1.0

def test_inference():
    #setup
    digits = datasets.load_digits()
    model = train_model(digits.data, digits.target)
    #assert for metric outputs
    pred = inference(model, digits.data)
    assert all(pred==digits.target)

