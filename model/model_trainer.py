import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelTrainer:
    def __init__(self, df, features, target):
        """Inisialisasi dengan DataFrame, fitur, dan target."""
        self.df = df
        self.features = features
        self.target = target
        self.model = RandomForestRegressor(random_state=42)
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def prepare_data(self, test_size=0.2):
        """Mempersiapkan data untuk training dan testing."""
        if not all(col in self.df.columns for col in self.features + [self.target]):
            missing = set(self.features + [self.target]) - set(self.df.columns)
            raise KeyError(f"[ERROR] Kolom berikut tidak ditemukan: {missing}")

        X = self.df[self.features]
        y = self.df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print("[INFO] Data training dan testing dipersiapkan.")

    def train(self):
        """Melatih model regresi."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("[ERROR] Data training belum disiapkan.")

        self.model.fit(self.X_train, self.y_train)
        print("[INFO] Model berhasil dilatih.")

    def evaluate(self):
        """Evaluasi performa model dan tampilkan metrik."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("[ERROR] Data testing belum disiapkan.")

        self.y_pred = self.model.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"R2 Score: {r2}")

        return mae, mse, r2

    def save_model(self, path='artifacts/model.pkl'):
        """Menyimpan model ke file pickle (.pkl)."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[INFO] Model disimpan ke: {path}")
