import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Memuat Data
file_path = r'D:\Backup\Kuliah\Skripsi\Machine Learning\FilterData\data\final_exoplanet_data.csv'
data = pd.read_csv(file_path)

# 2. Memilih Fitur dan Target
X = data[['semi_major_axis', 'star_teff', 'mag_v']].dropna()
y = data['habitable_zone_status'].loc[X.index]

# 3. Membagi Data menjadi Data Latih dan Uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normalisasi Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Menginisialisasi Model
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Neural Network': MLPClassifier(max_iter=500)
}

# 6. Melatih dan Menguji Setiap Model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi
    print(f"{name}:\n{classification_report(y_test, y_pred, zero_division=0)}")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.15f}\n")

# 7. Menggunakan Model untuk Prediksi Data Kandidat
candidate_data_path = r'D:\Backup\Kuliah\Skripsi\Machine Learning\FilterData\data\final_candidate_data.csv'  
candidate_data = pd.read_csv(candidate_data_path)

# 8. Pra-proses Data Kandidat
candidate_X = candidate_data[['semi_major_axis', 'star_teff', 'mag_v']].dropna()

# Normalisasi Data Kandidat
candidate_X = scaler.transform(candidate_X)

# 9. Melakukan Prediksi
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(candidate_X)

# 10. Menggabungkan hasil prediksi ke dalam DataFrame
for name in predictions:
    candidate_data[f'{name}_prediction'] = predictions[name]

# 11. Menyimpan hasil prediksi
output_candidate_path = r'D:\Backup\Kuliah\Skripsi\Machine Learning\FilterData\data\predicted_candidate_exoplanets.csv'
candidate_data.to_csv(output_candidate_path, index=False)
print(f"Predictions have been saved to {output_candidate_path}")

# 12. Jika ada label untuk data kandidat, hitung akurasi
if 'habitable_zone_status' in candidate_data.columns:
    candidate_y = candidate_data['habitable_zone_status']  # Pastikan kolom ini ada
    candidate_accuracy = {}
    for name, model in models.items():
        candidate_pred = predictions[name]
        candidate_accuracy[name] = accuracy_score(candidate_y, candidate_pred)

        print(f"{name} Candidate Accuracy: {candidate_accuracy[name]:.15f}\n")
else:
    print("No true labels found in candidate data for accuracy calculation.")
