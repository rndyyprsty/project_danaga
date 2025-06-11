import pandas as pd
import pickle

class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """Load model dari file pickle."""
        try:
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
            print("Model berhasil dimuat.")
        except FileNotFoundError:
            print(f"Model tidak ditemukan di: {self.model_path}")
        except Exception as e:
            print(f"Terjadi kesalahan saat memuat model: {e}")

    def predict(self, input_data: pd.DataFrame):
        """Melakukan prediksi dari input DataFrame."""
        if self.model is None:
            raise ValueError("Model belum dimuat. Panggil 'load_model()' terlebih dahulu.")
        
        try:
            prediction = self.model.predict(input_data)
            return prediction
        except Exception as e:
            print(f"Terjadi kesalahan saat prediksi: {e}")
            return None
