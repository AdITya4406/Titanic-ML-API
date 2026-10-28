
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Titanic ML API is running"


def test_predict():
    payload = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["survived"] in [0, 1]
    assert response.json()["result"] in [
        "Survived", "Did not survive"
    ]
