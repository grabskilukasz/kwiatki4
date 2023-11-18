import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3

app = FastAPI()
knc = KNeighborsClassifier(n_neighbors=3)


class IrisItem(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def root():
    return {"Hello": "World"}


@app.get("/train")
def train():
    df = pd.read_csv("Iris.csv")
    X = df.drop(["Id", "Species"], axis=1)
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    knc.fit(X_train, y_train)
    y_pred = knc.predict(X_test)
    acc_knn = metrics.accuracy_score(y_pred, y_test)
    print("The accuracy of the KNN is", acc_knn)

    with open("knn_model.pkl", "wb") as file:
        pickle.dump(knc, file)

    return f" The accuracy of the KNN is {acc_knn}"


@app.post("/predict")
def predict_species(item: IrisItem):
    try:
        # Przygotuj dane wejściowe dla modelu
        input_data = [[item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]]

        # Dokonaj predykcji przy użyciu wcześniej wytrenowanego modelu
        prediction = knc.predict(input_data)

        # Zwróć wynik predykcji
        return {"predicted_species": prediction[0]}

    except Exception as e:
        # W przypadku błędu zwróć informację o błędzie
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/respond")
def respond(number: int):
    response_message = f"Received number: {number}."

    conn = sqlite3.connect("example.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS numbers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            number INTEGER NOT NULL
        )
    """
    )
    cursor.execute("INSERT INTO numbers (number) VALUES (?)", (number,))
    conn.commit()
    conn.close()

    return {"response": response_message}


@app.get("/read")
def respond():
    conn = sqlite3.connect("example.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM numbers")
    results = cursor.fetchall()

    for row in results:
        print(row)

    return row
