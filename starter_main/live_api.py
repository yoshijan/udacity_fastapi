import requests

data = {
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
}
r = requests.post("https://udacity-fastapi.herokuapp.com/prediction", json=data)
print(r.status_code, r.reason)
print(r.text)