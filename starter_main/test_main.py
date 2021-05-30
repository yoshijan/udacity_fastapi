from fastapi.testclient import TestClient
from .main  import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_prediction_item():
    response = client.post(
        "/prediction/",
        json={
            "age": 42,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native-country": "United-States"
        },
    )
    assert response.status_code == 200
    assert response.json() ==  "[0]"


def test_score():
    response = client.post(
        "/slices_score/",
        json={
            "feature_list": [
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country"
            ]
        },
    )
    assert response.status_code == 200
    #model is always retrained so score varies
    #assert response.json() == "0.7127393838467944, 0.5320074580484773, 0.6092526690391459"
