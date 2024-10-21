import pandas as pd
import numpy as np

# Fungsi untuk menghitung magnitudo absolut dari magnitudo visual dan jarak
def calculate_absolute_magnitude(mag_v, distance):
    return mag_v - 5 * np.log10(distance) + 5

# Fungsi untuk menghitung luminositas dari magnitudo absolut
def calculate_luminosity_from_magnitude(abs_magnitude):
    M_sun = 4.83  # Magnitudo absolut matahari
    luminosity = 10 ** (0.4 * (M_sun - abs_magnitude))
    return luminosity

# Fungsi untuk memproses file CSV
def process_csv(file_path):
    # Membaca file CSV
    df = pd.read_csv(file_path)

    # Menyeleksi data: Menghapus baris yang memiliki nilai kosong (NaN)
    df_cleaned = df.dropna(subset=['semi_major_axis', 'star_distance', 'star_temperature', 'mag_v'])

    # Menghitung magnitudo absolut dan luminositas
    df_cleaned['absolute_magnitude'] = df_cleaned.apply(lambda row: calculate_absolute_magnitude(row['mag_v'], row['star_distance']), axis=1)
    df_cleaned['luminosity'] = df_cleaned['absolute_magnitude'].apply(calculate_luminosity_from_magnitude)

    # Menyimpan hasil ke file baru
    output_file = file_path.split(".csv")[0] + "_processed.csv"
    df_cleaned.to_csv(output_file, index=False)
    print(f"Data processed and saved to {output_file}")

# Fungsi utama
def main():
    # Ganti dengan path file CSV yang sesuai
    file_path = "D:/Backup/Kuliah/Skripsi/Machine Learning/FilterData/data/filterdata.csv"
    process_csv(file_path)

if __name__ == "__main__":
    main()
