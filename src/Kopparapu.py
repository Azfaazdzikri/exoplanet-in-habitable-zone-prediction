import pandas as pd
import numpy as np

# From Kopparapu et al. 2014: Calculate Stellar Flux
def flux(luminosity, semi_major_axis):
    return (luminosity / (semi_major_axis ** 2))

# Kopparapu calculation to determine Seff boundaries based on temperature delta
def kopparapu_boundary(temp_delta, SeffSUN, a, b, c, d):
    return SeffSUN + a * temp_delta + b * (temp_delta ** 2) + c * (temp_delta ** 3) + d * (temp_delta ** 4)

# Calculate habitable zone boundaries based on star effective temperature
def calculate_hz_boundaries(star_teff):
    tS = star_teff - 5780  # Temperature delta from 5780 K
    zones = {
        'recentVenus': kopparapu_boundary(tS, 1.766, 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15),
        'runawayGreenhouse': kopparapu_boundary(tS, 1.107, 1.332e-4, 1.580e-8, -8.308e-12, -1.931e-15),
        'maximumGreenhouse': kopparapu_boundary(tS, 0.356, 6.171e-5, 1.689e-9, -3.198e-12, -5.575e-16),
        'earlyMars': kopparapu_boundary(tS, 0.320, 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16)
    }
    return zones

# Determine if planet is in habitable zone and calculate fluxes
def check_habitable_zone(df):
    # Initialize new columns
    df['habitable_zone_status'] = ''
    df['HZ_distance_recent_venus'] = np.nan
    df['HZ_distance_runaway_greenhouse'] = np.nan
    df['HZ_distance_maximum_greenhouse'] = np.nan
    df['HZ_distance_early_mars'] = np.nan
    df['stellar_flux_recent_venus'] = np.nan
    df['stellar_flux_runaway_greenhouse'] = np.nan
    df['stellar_flux_maximum_greenhouse'] = np.nan
    df['stellar_flux_early_mars'] = np.nan

    for index, row in df.iterrows():
        if pd.isna(row['semi_major_axis']) or pd.isna(row['star_teff']) or pd.isna(row['luminosity']):
            continue

        star_teff = row['star_teff']
        semi_major_axis = row['semi_major_axis']
        luminosity = row['luminosity']

        # Calculate the boundaries of habitable zones
        boundaries = calculate_hz_boundaries(star_teff)
        hz_distances = {
            'recentVenus': (luminosity / boundaries['recentVenus']) ** 0.5,
            'runawayGreenhouse': (luminosity / boundaries['runawayGreenhouse']) ** 0.5,
            'maximumGreenhouse': (luminosity / boundaries['maximumGreenhouse']) ** 0.5,
            'earlyMars': (luminosity / boundaries['earlyMars']) ** 0.5
        }

        # Store distances
        df.at[index, 'HZ_distance_recent_venus'] = hz_distances['recentVenus'].real
        df.at[index, 'HZ_distance_runaway_greenhouse'] = hz_distances['runawayGreenhouse'].real
        df.at[index, 'HZ_distance_maximum_greenhouse'] = hz_distances['maximumGreenhouse'].real
        df.at[index, 'HZ_distance_early_mars'] = hz_distances['earlyMars'].real

        df.at[index, 'stellar_flux_recent_venus'] = flux(luminosity, hz_distances['recentVenus'].real)
        df.at[index, 'stellar_flux_runaway_greenhouse'] = flux(luminosity, hz_distances['runawayGreenhouse'].real)
        df.at[index, 'stellar_flux_maximum_greenhouse'] = flux(luminosity, hz_distances['maximumGreenhouse'].real)
        df.at[index, 'stellar_flux_early_mars'] = flux(luminosity, hz_distances['earlyMars'].real)
        planet_flux = flux(luminosity, semi_major_axis)

        # Determine habitable zone status
        if planet_flux < boundaries['earlyMars']:
            hz_status = "Not in HZ"
        elif boundaries['earlyMars'] <= planet_flux <= boundaries['maximumGreenhouse']:
            hz_status = "Optimistic HZ"
        elif boundaries['maximumGreenhouse'] <= planet_flux <= boundaries['runawayGreenhouse']:
            hz_status = "Conservative HZ"
        else:
            hz_status = "Not in HZ"

        df.at[index, 'habitable_zone_status'] = hz_status

    return df

# Example usage
file_path = r'D:\Backup\Kuliah\Skripsi\Machine Learning\exoplanet-in-habitable-zone-prediciton\data\candidate_exoplanet_aschwaden.csv'
df = pd.read_csv(file_path)
df.dropna(subset=['semi_major_axis', 'star_teff', 'luminosity'], inplace=True)

# Check habitable zone and calculate stellar flux
df = check_habitable_zone(df)

# Output processed data
output_file_path = r'D:\Backup\Kuliah\Skripsi\Machine Learning\exoplanet-in-habitable-zone-prediciton\data\candidate_exoplanet_kopparapu.csv'
df.to_csv(output_file_path, index=False)
print(f"Data has been processed and saved to {output_file_path}")
