import pandas as pd
import numpy as np

# Fungsi untuk menghitung luminositas dari mag_v dan star_distance
def calculate_luminosity(mag_v, star_distance):
    if pd.notnull(mag_v) and pd.notnull(star_distance):
        # Hitung luminositas berdasarkan rumus
        luminosity = 10 ** (0.4 * (4.83 - (mag_v - 5 * np.log10(star_distance) + 5)))
        return luminosity
    else:
        return np.nan  # Jika ada nilai NaN, kembalikan NaN

# Fungsi utama untuk filter dan pengolahan data
def process_exoplanet_data(file_path):
    # Baca data dari file CSV
    data = pd.read_csv(file_path)

    # Filter data hanya untuk sistem keplanetan dengan minimal 2 planet
    # Gunakan 'star_name' untuk mengelompokkan planet dalam satu sistem bintang
    planet_count_per_system = data['star_name'].value_counts()
    systems_with_multiple_planets = planet_count_per_system[planet_count_per_system >= 2].index
    filtered_data = data[data['star_name'].isin(systems_with_multiple_planets)]

    # Mengisi nilai kosong dengan rata-rata kolom terkait
    filtered_data['orbital_period'].fillna(filtered_data['orbital_period'].mean(), inplace=True)
    filtered_data['semi_major_axis'].fillna(filtered_data['semi_major_axis'].mean(), inplace=True)
    filtered_data['mag_v'].fillna(filtered_data['mag_v'].mean(), inplace=True)
    filtered_data['star_teff'].fillna(filtered_data['star_teff'].mean(), inplace=True)
    filtered_data['star_distance'].fillna(filtered_data['star_distance'].mean(), inplace=True)

    # Hitung luminositas untuk setiap baris
    filtered_data['luminosity'] = filtered_data.apply(
        lambda row: calculate_luminosity(row['mag_v'], row['star_distance']), axis=1)

    # Menyimpan hasil ke file CSV baru
    output_file_path = r'D:\Backup\Kuliah\Skripsi\Machine Learning\FilterData\data\filterdata_candidate_processed.csv'
    filtered_data.to_csv(output_file_path, index=False)
    print(f"Data has been processed and saved to {output_file_path}")

# Path ke file CSV yang akan diolah
file_path = r'D:\Backup\Kuliah\Skripsi\Machine Learning\FilterData\data\filterdata_candidate.csv'
process_exoplanet_data(file_path)
