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

        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sepal_length_t FLOAT,
                sepal_width_t FLOAT,
                petal_length_t FLOAT,
                petal_width_t FLOAT,
                prediction_t STRING
            )
        """
        )
        cursor.execute(
            "INSERT INTO predictions (sepal_length_t, sepal_width_t, petal_length_t, petal_width_t, prediction_t) VALUES (?, ?, ?, ?, ?)",
            (
                (item.sepal_length),
                (item.sepal_width),
                (item.petal_length),
                (item.petal_width),
                (prediction[0]),
            ),
        )
        last_id = cursor.lastrowid

        conn.commit()
        conn.close()

        # Zwróć wynik predykcji
        return f"predicted_species: {prediction[0]}, ID:{last_id}"

    except Exception as e:
        # W przypadku błędu zwróć informację o błędzie
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/read")
def read(id):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM predictions WHERE id = ?", (id,))
    results = cursor.fetchone()

    return results



# @app.post("/respond")
# def respond(number: int):
#     response_message = f"Received number: {number}."
#
#     conn = sqlite3.connect("example.db")
#     cursor = conn.cursor()
#     cursor.execute(
#         """
#         CREATE TABLE IF NOT EXISTS numbers (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             number INTEGER NOT NULL
#         )
#     """
#     )
#     cursor.execute("INSERT INTO numbers (number) VALUES (?)", (number,))
#     conn.commit()
#     conn.close()
#
#     return {"response": response_message}


