import pandas as pd
import numpy as np
import string

# Function to get the next planet letter
def get_next_letter(existing_planets):
    if existing_planets:
        last_letter = existing_planets[-1]
        if last_letter == 'z':
            next_letter = 'aa'
        elif len(last_letter) == 1:
            next_letter = string.ascii_lowercase[string.ascii_lowercase.index(last_letter) + 1]
        else:
            next_letter = chr(ord(last_letter[0]) + 1) + last_letter[1]
    else:
        next_letter = 'b'

    return next_letter

# Function to predict new exoplanets based on harmonic ratios
def predict_new_exoplanets(df):
    harmonic_ratios = [(2, 1), (3, 2), (5, 3), (5, 4), (4, 3), (5, 2), (3, 1)]
    new_rows = []

    for star, group in df.groupby('star_name'):
        periods = group['orbital_period'].dropna().sort_values().values
        if len(periods) < 2:
            continue

        existing_planets = group['planet_name'].str.extract(r'([bcdefghijklmnopqrstuvwxyz]+)$')[0].dropna().tolist()
        next_letter = get_next_letter(existing_planets)

        for i in range(len(periods) - 1):
            for ratio in harmonic_ratios:
                predicted_period = periods[i] * ratio[0] / ratio[1]
                if predicted_period not in periods:
                    # Use Kepler's Third Law to estimate semi-major axis for the predicted planet
                    star_mass = 1  # Assuming 1 solar mass, adjust if data is available
                    predicted_semi_major_axis = (predicted_period**(2/3)) * (star_mass**(1/3))

                    new_row = {
                        'planet_name': f"{star} {next_letter}",
                        'planet_status': 'Aschwaden Prediction',
                        'orbital_period': predicted_period,
                        'semi_major_axis': predicted_semi_major_axis,
                        'star_name': star,
                        'mag_v': group['mag_v'].iloc[0],
                        'star_teff': group['star_teff'].iloc[0],
                        'star_distance': group['star_distance'].iloc[0],
                        'harmonic_ratio': f"{ratio[0]} banding {ratio[1]}"  # Use text description for harmonic ratio
                    }
                    new_rows.append(new_row)
                    next_letter = get_next_letter([next_letter])

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)

    return df

# Function to calculate habitable zone boundaries
def calculate_hz_boundaries(star_teff):
    tS = star_teff - 5780
    zones = {
        'recentVenus': (1.766, 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15),
        'runawayGreenhouse': (1.107, 1.332e-4, 1.580e-8, -8.308e-12, -1.931e-15),
        'maximumGreenhouse': (0.356, 6.171e-5, 1.689e-9, -3.198e-12, -5.575e-16),
        'earlyMars': (0.320, 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16)
    }

    def kopparapu2014(SeffSUN, a, b, c, d):
        return SeffSUN + a * tS + b * tS**2 + c * tS**3 + d * tS**4

    boundaries = {}
    for zone, params in zones.items():
        boundaries[zone] = kopparapu2014(*params)

    return boundaries

# Function to check if exoplanet is in habitable zone
def check_habitable_zone(df):
    for index, row in df.iterrows():
        if pd.isna(row['semi_major_axis']) or pd.isna(row['star_teff']):
            continue

        star_teff = row['star_teff']
        semi_major_axis = row['semi_major_axis']

        boundaries = calculate_hz_boundaries(star_teff)

        if semi_major_axis < boundaries['earlyMars']:
            hz_status = "Not in HZ"
        elif boundaries['earlyMars'] <= semi_major_axis <= boundaries['maximumGreenhouse']:
            hz_status = "Optimistic HZ"
        elif boundaries['maximumGreenhouse'] <= semi_major_axis <= boundaries['runawayGreenhouse']:
            hz_status = "Conservative HZ"
        else:
            hz_status = "Not in HZ"

        df.at[index, 'habitable_zone_status'] = hz_status

    return df

# Main processing function
def process_exoplanet_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(subset=['orbital_period', 'semi_major_axis', 'star_teff', 'mag_v', 'star_distance'], inplace=True)

    df = predict_new_exoplanets(df)

    df = check_habitable_zone(df)

    # Convert 'harmonic_ratio' to string before writing to CSV to prevent unwanted formatting
    df['harmonic_ratio'] = df['harmonic_ratio'].astype(str)

    output_file_path = r'D:\Backup\Kuliah\Skripsi\Machine Learning\FilterData\data\final_exoplanet_data.csv'
    df.to_csv(output_file_path, index=False, date_format='%Y-%m-%d')
    print(f"Data has been processed and saved to {output_file_path}")

# Example usage
file_path = r'D:\Backup\Kuliah\Skripsi\Machine Learning\FilterData\data\filtered_exoplanet_data.csv'
process_exoplanet_data(file_path)
