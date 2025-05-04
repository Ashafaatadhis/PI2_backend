from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib # Untuk memuat scaler

# Memuat model Keras yang telah dilatih
model = tf.keras.models.load_model("model_tf.keras")

# Memuat objek scaler yang telah di-fit
scaler = joblib.load('scaler.joblib')

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Mendefinisikan skema input data menggunakan Pydantic
class StressInput(BaseModel):
    jam_tidur: float
    screen_time: float
    waktu_olahraga: float
    waktu_belajar: float
    jumlah_tugas: int 

# Membuat endpoint untuk prediksi (/predict) dengan metode POST
@app.post("/predict")
def predict_stress(data: StressInput):
    # 1. Konversi data input Pydantic ke numpy array
    input_data = np.array([[
        data.jam_tidur,
        data.screen_time,
        data.waktu_olahraga,
        data.waktu_belajar,
      
        float(data.jumlah_tugas)
    ]])

    # 2. Lakukan pra-pemrosesan (Normalisasi/Scaling) pada data input
    # Gunakan scaler yang SAMA dengan saat training
    input_scaled = scaler.transform(input_data)

    # 3. Melakukan prediksi menggunakan model pada data yang sudah di-scale
    prediction_probs = model.predict(input_scaled)

    # 4. Mengambil label kelas dengan probabilitas tertinggi
    predicted_label = int(np.argmax(prediction_probs))

    # 5. Mengambil probabilitas prediksi untuk setiap kelas (confidence)
    confidence_scores = prediction_probs[0].tolist() # Ambil dari batch pertama

    # Mengembalikan hasil prediksi dalam format JSON
    return {
        "tingkat_stres": predicted_label,
        "confidence": confidence_scores,
    }

 