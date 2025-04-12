from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("model_tf.keras")

app = FastAPI()

# Schema input sesuai kolom
class StressInput(BaseModel):
    jam_tidur: float
    screen_time: float
    waktu_olahraga: float
    waktu_belajar: float
    jumlah_tugas: int

@app.post("/predict")
def predict_stress(data: StressInput):
    input_array = np.array([[
        data.jam_tidur,
        data.screen_time,
        data.waktu_olahraga,
        data.waktu_belajar,
        data.jumlah_tugas
    ]])

    prediction = model.predict(input_array)
    predicted_label = int(np.argmax(prediction))

    return {
        "tingkat_stres": predicted_label,
        "confidence": prediction.tolist()[0]
    }
