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

# Fungsi untuk menghitung fluks bintang
def flux(luminosity, semi):
    return ((1 / semi)**2) * luminosity  # Fluks bintang

# Fungsi untuk menghitung batas zona layak huni
def getSeffBoundary(temp, zone):
    tS = temp - 5780  # Selisih temperatur
    if zone == "recentVenus":
        SeffSUN = 1.766
        a = 2.136 * (10**-4)
        b = 2.533 * (10**-8)
        c = -1.332 * (10**-11)
        d = -3.097 * (10**-15)
    elif zone == "runawayGreenhouse":
        SeffSUN = 1.107
        a = 1.332 * (10**-4)
        b = 1.580 * (10**-8)
        c = -8.308 * (10**-12)
        d = -1.931 * (10**-15)
    elif zone == "maximumGreenhouse":
        SeffSUN = 0.356
        a = 6.171 * (10**-5)
        b = 1.689 * (10**-9)
        c = -3.198 * (10**-12)
        d = -5.575 * (10**-16)
    elif zone == "earlyMars":
        SeffSUN = 0.320
        a = 5.547 * (10**-5)
        b = 1.526 * (10**-9)
        c = -2.874 * (10**-12)
        d = -5.011 * (10**-16)

    return Kopparapu2014(SeffSUN, a, b, c, d, tS)

# Fungsi dari Kopparapu 2014
def Kopparapu2014(SeffSUN, a, b, c, d, tS):
    return SeffSUN + a * tS + b * (tS ** 2) + c * (tS ** 3) + d * (tS ** 4)

# Fungsi untuk memproses file CSV
def process_csv(file_path):
    # Membaca file CSV
    df = pd.read_csv(file_path)

    # Menyeleksi data: Menghapus baris yang memiliki nilai kosong (NaN)
    df_cleaned = df.dropna(subset=['semi_major_axis', 'star_distance', 'star_temperature', 'mag_v'])

    # Menghitung magnitudo absolut dan luminositas
    df_cleaned['absolute_magnitude'] = df_cleaned.apply(lambda row: calculate_absolute_magnitude(row['mag_v'], row['star_distance']), axis=1)
    df_cleaned['luminosity'] = df_cleaned['absolute_magnitude'].apply(calculate_luminosity_from_magnitude)

    # Menghitung HZ
    for index, row in df_cleaned.iterrows():
        star_temp = row['star_temperature']
        luminosity = row['luminosity']
        semi_major_axis = row['semi_major_axis']

        recentVenus = getSeffBoundary(star_temp, "recentVenus")
        runawayGreenhouse = getSeffBoundary(star_temp, "runawayGreenhouse")
        maximumGreenhouse = getSeffBoundary(star_temp, "maximumGreenhouse")
        earlyMars = getSeffBoundary(star_temp, "earlyMars")

        stellar_flux = flux(luminosity, semi_major_axis)

        # Menentukan status HZ
        if stellar_flux < earlyMars:
            df_cleaned.at[index, 'HZ_status'] = "Not in HZ"
        elif earlyMars <= stellar_flux <= maximumGreenhouse:
            df_cleaned.at[index, 'HZ_status'] = "Optimistic HZ"
        elif maximumGreenhouse < stellar_flux <= runawayGreenhouse:
            df_cleaned.at[index, 'HZ_status'] = "Conservative HZ"
        elif runawayGreenhouse < stellar_flux <= recentVenus:
            df_cleaned.at[index, 'HZ_status'] = "Optimistic HZ"
        else:
            df_cleaned.at[index, 'HZ_status'] = "Not in HZ"

    # Menyimpan hasil ke file baru
    output_file = file_path.split(".csv")[0] + "_HZ.csv"
    df_cleaned.to_csv(output_file, index=False)
    print(f"Data processed and saved to {output_file}")

# Fungsi utama
def main():
    # Ganti dengan path file CSV yang sesuai
    file_path = "D:/Backup/Kuliah/Skripsi/Machine Learning/FilterData/data/filterdata_processed.csv"
    process_csv(file_path)

if __name__ == "__main__":
    main()
