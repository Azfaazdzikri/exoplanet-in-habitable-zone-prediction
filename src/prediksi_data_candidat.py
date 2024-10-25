import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.impute import SimpleImputer

# Load the candidate and training data
candidate_data = pd.read_csv(r'D:\Azfa\Kuliah\Semester 7\skripsi\ML\exoplanet-in-habitable-zone-prediction\data\final_candidate_data.csv')
train_data = pd.read_csv(r'D:\Azfa\Kuliah\Semester 7\skripsi\ML\exoplanet-in-habitable-zone-prediction\data\final_exoplanet_data.csv')

# Define features and targets
features = [
    'semi_major_axis',
    'star_teff',
    'mag_v',
    'orbital_period',
    'luminosity'
]

# Target untuk prediksi jarak habitable zone
targets_hz_distance = [
    'HZ_distance_recent_venus',
    'HZ_distance_runaway_greenhouse',
    'HZ_distance_maximum_greenhouse',
    'HZ_distance_early_mars'
]

# Target untuk status habitable zone
target_hz_status = 'habitable_zone_status'

# Prepare training data
X = train_data[features]
y_distance = train_data[targets_hz_distance]
y_status = train_data[target_hz_status]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
candidate_data[features] = imputer.transform(candidate_data[features])

# Split training data for distance predictions
X_train, X_test, y_train_distance, y_test_distance = train_test_split(X, y_distance, test_size=0.2, random_state=42)

# Model untuk jarak habitable zone
models_distance = {
    'Random Forest': RandomForestRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'K-Neighbors': KNeighborsRegressor(),
    'SVM': SVR(),
    'Linear Regression': LinearRegression()
}

# Tambahkan Polynomial Regression
poly = PolynomialFeatures(degree=2)  # Anda bisa mengubah derajat sesuai kebutuhan
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
models_distance['Polynomial Regression'] = LinearRegression()

# Simpan prediksi untuk jarak habitable zone
predictions_distance = {}
for target in targets_hz_distance:
    for name, model in models_distance.items():
        if name == 'Polynomial Regression':
            model.fit(X_poly_train, y_train_distance[target])
            predictions_distance[f'{target}_{name}'] = model.predict(poly.transform(candidate_data[features]))
        else:
            model.fit(X_train, y_train_distance[target])
            predictions_distance[f'{target}_{name}'] = model.predict(candidate_data[features])

        # Hitung metrik untuk data uji
        y_pred_distance = model.predict(X_poly_test) if name == 'Polynomial Regression' else model.predict(X_test)
        r2 = r2_score(y_test_distance[target], y_pred_distance)
        rmse = np.sqrt(mean_squared_error(y_test_distance[target], y_pred_distance))
        mae = mean_absolute_error(y_test_distance[target], y_pred_distance)

        print(f'{target} Metrics for {name}:')
        print(f' - R-Squared: {r2}')
        print(f' - RMSE: {rmse}')
        print(f' - MAE: {mae}\n')

# Prepare for habitable zone status prediction
X_train_status, X_test_status, y_train_status, y_test_status = train_test_split(X, y_status, test_size=0.2, random_state=42)

# Model untuk status habitable zone
models_status = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Neighbors': KNeighborsClassifier(),
    'SVM': SVC()
}

# Simpan prediksi untuk status habitable zone
for name, model in models_status.items():
    model.fit(X_train_status, y_train_status)

    # Prediksi status habitable zone untuk data kandidat
    predictions_status = model.predict(candidate_data[features])

    # Hitung metrik untuk klasifikasi
    accuracy = accuracy_score(y_test_status, model.predict(X_test_status))
    f1 = f1_score(y_test_status, model.predict(X_test_status), average='weighted')
    recall = recall_score(y_test_status, model.predict(X_test_status), average='weighted')
    precision = precision_score(y_test_status, model.predict(X_test_status), average='weighted')

    print(f'Habitable Zone Status Classification Metrics for {name}:')
    print(f' - Accuracy: {accuracy}')
    print(f' - F1 Score: {f1}')
    print(f' - Recall: {recall}')
    print(f' - Precision: {precision}\n')

    # Simpan prediksi ke dalam kolom terpisah di candidate_data
    candidate_data[f'predicted_habitable_zone_status_{name}'] = predictions_status

# Tambahkan semua prediksi ke dalam data kandidat
for target in targets_hz_distance:
    for model_name in models_distance.keys():
        candidate_data[f'HZ_distance_{target}_{model_name}_prediction'] = predictions_distance[f'{target}_{model_name}']

# Simpan hasil prediksi ke file baru
candidate_data.to_csv(r'D:\Azfa\Kuliah\Semester 7\skripsi\ML\exoplanet-in-habitable-zone-prediction\data\predicted_candidate_exoplanets.csv', index=False)
