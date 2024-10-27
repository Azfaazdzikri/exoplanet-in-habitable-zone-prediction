import pandas as pd
import numpy as np

def calculate_luminosity(mag_v):
    M_sun = 4.83  # Magnitudo absolut matahari
    return 10 ** ((M_sun - mag_v) / 2.5)

def get_next_letter(existing_planets):
    existing_letters = [planet[-1] for planet in existing_planets]
    next_letter = 'b'

    while next_letter in existing_letters:
        next_letter = chr(ord(next_letter) + 1)  # Increment to next letter

    return next_letter

def is_planetary_system_valid(group):
    if len(group) < 2:
        return False  # Hapus sistem dengan hanya 1 planet

    distances = group['semi_major_axis'].diff().dropna().abs()
    return (distances <= 0.5).any()  # Cek apakah ada planet dengan jarak dekat

def filter_data(df):
    # Menghitung rata-rata untuk kolom numerik yang kosong
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Memfilter sistem keplanetan yang valid
    valid_systems = df.groupby('star_name').filter(is_planetary_system_valid)
    return valid_systems

def predict_new_exoplanets(df):
    harmonic_ratios = [(2, 1), (3, 2), (5, 3), (5, 4), (4, 3), (5, 2), (3, 1)]
    new_rows = []

    for star, group in df.groupby('star_name'):
        periods = group['orbital_period'].dropna().sort_values().values
        semi_major_axes = group['semi_major_axis'].dropna().sort_values().values
        planet_names = group['planet_name'].tolist()
        star_distances = group['star_distance'].tolist()  # Mengambil nilai star_distance

        existing_planets = group['planet_name'].str.extract(r'([bcdefghijklmnopqrstuvwxyz]+)$')[0].dropna().tolist()
        next_letter = get_next_letter(existing_planets)

        for i in range(len(periods) - 1):
            for ratio in harmonic_ratios:
                predicted_period = periods[i] * ratio[0] / ratio[1]
                if predicted_period not in periods:
                    predicted_semi_major_axis = semi_major_axes[i] * (predicted_period / periods[i])**(2/3)

                    # Ambil magnitudo dan teff dari bintang
                    mag_v = group['mag_v'].iloc[0]

                    # Ambil nama planet referensi untuk harmonic_ratio
                    referenced_planet = planet_names[i]
                    harmonic_ratio = f"{ratio[0]} to {ratio[1]}"

                    # Ambil star_distance dari planet yang dirujuk
                    referenced_star_distance = star_distances[i]

                    new_row = {
                        'planet_name': f"{star} {next_letter}",
                        'planet_status': 'Aschwaden Prediction',
                        'referenced_planet': referenced_planet,
                        'orbital_period': predicted_period,
                        'semi_major_axis': predicted_semi_major_axis,
                        'star_name': star,
                        'mag_v': mag_v,
                        'star_teff': group['star_teff'].iloc[0],
                        'star_distance': referenced_star_distance,  # Menyertakan star_distance yang dirujuk
                        'harmonic_ratio': harmonic_ratio  # Menyertakan kolom harmonic_ratio
                    }
                    new_rows.append(new_row)
                    next_letter = get_next_letter([next_letter])

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)

    return df

# Contoh penggunaan
file_path = r'D:\Backup\Kuliah\Skripsi\Machine Learning\exoplanet-in-habitable-zone-prediciton\data\candidate_exoplanet_data.csv'
df = pd.read_csv(file_path)

# Memfilter data
df_filtered = filter_data(df)

# Memperoleh data eksoplanet baru
df_with_predictions = predict_new_exoplanets(df_filtered)

# Menghitung luminositas untuk semua planet dalam dataframe
df_with_predictions['luminosity'] = df_with_predictions['mag_v'].apply(calculate_luminosity)

# Menyimpan data yang diproses
output_file_path = r'D:\Backup\Kuliah\Skripsi\Machine Learning\exoplanet-in-habitable-zone-prediciton\data\candidate_exoplanet_aschwaden.csv'
df_with_predictions.to_csv(output_file_path, index=False)
print(f"Data telah diproses dan disimpan ke {output_file_path}")
