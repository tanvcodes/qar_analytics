import os
import pandas as pd
import numpy as np
from math import sin, cos, pi, radians, atan2, sqrt


def get_dist(row):
    r = 6373.0

    # Converting to radians
    lat1 = radians(row['latitude_deg'])
    lon1 = radians(row['longitude_deg'])
    lat2 = radians(row['Lat2'])
    lon2 = radians(row['Lon2'])

    # Calculating delta between lat and long
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Calculating the distance between two points
    aa = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(aa), sqrt(1 - aa))
    distance = r * c
    return distance


CSVDir = 'C:\\Users\\tanveer\\Documents\\Work\\Proj\\Tail652\\CSV\\'
table_dir = 'C:\\Users\\tanveer\\Documents\\Work\\Proj\\Tail652\\Table\\'
airport_dir = 'C:\\Users\\tanveer\\Documents\\Work\\Proj\\airports.csv'
runway_dir = 'C:\\Users\\tanveer\\Documents\\Work\\Proj\\runways.csv'

airports = pd.read_csv(airport_dir)
runways = pd.read_csv(runway_dir)
f_list = os.listdir(CSVDir)
f_list.sort()
f_list = [x for x in f_list if 'meta' not in x]
n = len(f_list)
datapoint_dict = {}
not_compiled_list = {}

table1 = pd.DataFrame(
    columns=['flightNum', 'CAS_touchdown', 'PITCH_touchdown', 'IVV_touchdown', 'Vert_accel_touchdown', 'AOA_touchdown',
             'PLA_touchdown', 'Vert_accel_landing', 'AOA_1s_before', 'AOA_8s_before', 'Pitch_angle_1s_before', 'Pitch_angle_8s_before',
             'PLA_1s_before', 'PLA_8s_before', 'HEADWIND', 'XWIND', 'FQTY_sum_5s_before', 'TAS_1s_before', 'TAS_8s_before',
             'GS_1s_before', 'GS_8s_before', 'FLAP_8s_before', 'SAT_8s_before', 'ALT_10s_after', 'IVV_1s_before', 'IVV_8s_before',
             'ALTR_1s_before', 'ALTR_8s_before', 'Heading_landing', 'ldg_apt_id', 'ldg_apt_ident', 'ldg_apt_elevation_ft', 'ldg_apt_rwy'])

for i in range(0, n):
    print('working on ', f_list[i])
    f_name = CSVDir + f_list[i]
    flight_df = pd.read_csv(f_name, parse_dates=['UTC'])
    flight_df['UTC'] = flight_df['UTC'].dt.round('s')

    # Finding index where touchdown occurs
    ldg_idx = flight_df[flight_df['WOW'] == True].last_valid_index()

    # Checking if landing index is None, skipping those flights
    if ldg_idx is None:
        print('landing index is none, skipping this file')
        not_compiled_list[f_list[i]] = 'ldg_idx is None'
        continue
    datapoint_dict['LAT_ldg'] = flight_df['LATP'][ldg_idx]
    datapoint_dict['LON_ldg'] = flight_df['LONP'][ldg_idx]

    # Checking if CSV file somehow don't have data after landing. (happens only for 1 flight)
    if flight_df['WOW'].last_valid_index() == ldg_idx:
        print('last index is the landing index, skipping this file')
        not_compiled_list[f_list[i]] = 'ldg_idx is last index'
        continue
    ldg_threshold_idx = ldg_idx - 12

    a = flight_df[flight_df['MSQT_1'] == True].last_valid_index()
    b = flight_df[flight_df['MSQT_2'] == True].last_valid_index()

    # Checking if SQUAT switch gets switched on or not
    if a is None or b is None:
        not_compiled_list[f_list[i]] = 'touchdown idx is None'
        print('Touchdown index is None, skipping this file')
        continue
    touchdown_idx = max(a, b)

    datapoint_dict['flightNum'] = str(f_list[i][:-4])
    datapoint_dict['LAT_ldg'] = flight_df['LATP'][ldg_idx]
    datapoint_dict['LON_ldg'] = flight_df['LONP'][ldg_idx]

    datapoint_dict['CAS_touchdown'] = flight_df['CAS'][touchdown_idx]
    datapoint_dict['PITCH_touchdown'] = flight_df['PTCH'][touchdown_idx]
    datapoint_dict['IVV_touchdown'] = flight_df['IVV'][touchdown_idx]
    datapoint_dict['Vert_accel_touchdown'] = flight_df['VRTG'][touchdown_idx]
    datapoint_dict['AOA_touchdown'] = flight_df['AOAI'][touchdown_idx]
    datapoint_dict['PLA_touchdown'] = flight_df.loc[touchdown_idx][['PLA_1', 'PLA_2', 'PLA_3', 'PLA_4']].mean()

    # Calling function to extract vertical accel at touchdown
    datapoint_dict['Vert_accel_landing'] = flight_df['VRTG'][ldg_idx]

    # Task10: Angle of attack 1s and 8s before landing
    # Calling function to extract angle of attack 1s and 8s before touchdown
    datapoint_dict['AOA_1s_before'] = flight_df['AOAI'][ldg_idx - 1]
    datapoint_dict['AOA_8s_before'] = flight_df['AOAI'][ldg_idx - 8]

    # Task11: Pitch angle 1s and 8s before landing
    # Calling function to extract pitch angle 1s and 8s before landing
    datapoint_dict['Pitch_angle_1s_before'] = flight_df['PTCH'][ldg_idx - 1]
    datapoint_dict['Pitch_angle_8s_before'] = flight_df['PTCH'][ldg_idx - 8]

    # Task13: PLA 1s and 8s before landing
    # Extracting power lever angles 1s and 8s before landing
    PLA_1s, PLA_8s = [], []
    for j in range(1, 5):
        col = 'PLA_' + str(j)
        PLA_1s.append(flight_df[col][ldg_idx - 1])
        PLA_8s.append(flight_df[col][ldg_idx - 8])
    datapoint_dict['PLA_1s_before'] = sum(PLA_1s) / len(PLA_1s)
    datapoint_dict['PLA_8s_before'] = sum(PLA_8s) / len(PLA_8s)

    # Task12: Wind speeds at 50ft before landing
    # Extracting Tailwind and crosswind 50ft before landing
    df = pd.read_csv(f_name)
    fifty_ft_idx = df[(np.ceil(df['RALT'] / 10)) * 10 == 50.0].last_valid_index()

    # Checking if aircraft attains height of 50ft
    if fifty_ft_idx is None:
        fifty_ft_idx = df[(np.ceil(df['RALT'] / 10)) * 10 == 60.0].last_valid_index()
    if fifty_ft_idx is None:
        print('fifty ft index does not exist, skipping this file', f_list[i])
        not_compiled_list[f_list[i]] = 'fifty feet idx was None'
        continue
    datapoint_dict['HEADWIND'] = (df['WS'] * ((df['TH'] - df['WD']).apply(lambda x: cos(pi * x / 180))))[fifty_ft_idx]
    datapoint_dict['XWIND'] = (df['WS'] * ((df['TH'] - df['WD']).apply(lambda x: sin(pi * x / 180))))[fifty_ft_idx]

    # Task 1: Aircraft weight 5s before landing
    datapoint_dict['FQTY_sum_5s_before'] = flight_df.loc[ldg_idx - 5][['FQTY_1', 'FQTY_2', 'FQTY_3', 'FQTY_4']].sum()

    # Task2: airspeed 1s and 8s before landing
    datapoint_dict['TAS_1s_before'] = flight_df['TAS'][ldg_idx - 1]
    datapoint_dict['TAS_8s_before'] = flight_df['TAS'][ldg_idx - 8]

    # Task3: GroundSpeed 1s and 8s before landing
    datapoint_dict['GS_1s_before'] = flight_df['GS'][ldg_idx - 1]
    datapoint_dict['GS_8s_before'] = flight_df['GS'][ldg_idx - 8]

    # Task4: Landing flap setting 8s before landing
    datapoint_dict['FLAP_8s_before'] = flight_df['FLAP'][ldg_idx - 8]

    # Task7: Temperature 8s before landing
    datapoint_dict['SAT_8s_before'] = flight_df['SAT'][ldg_idx - 8]

    # Task8: landing altitude
    # code for airport elevation (based on a lookup table)
    datapoint_dict['ALT_10s_after'] = flight_df['ALT'][ldg_idx + 10]

    # Task9: Descent Rate
    datapoint_dict['IVV_1s_before'] = flight_df['IVV'][ldg_idx - 1]
    datapoint_dict['IVV_8s_before'] = flight_df['IVV'][ldg_idx - 8]

    datapoint_dict['ALTR_1s_before'] = flight_df['ALTR'][ldg_idx - 1]
    datapoint_dict['ALTR_8s_before'] = flight_df['ALTR'][ldg_idx - 8]

    datapoint_dict['Heading_landing'] = flight_df['TH'][ldg_idx]
    # to determine the destination airports & runways in use
    ldg_lat = flight_df['LATP'][ldg_idx]
    ldg_lon = flight_df['LONP'][ldg_idx]
    ldg_threshold_lat = flight_df['LATP'][ldg_threshold_idx]
    ldg_threshold_lon = flight_df['LONP'][ldg_threshold_idx]

    # to determine the destination airportid and identifiers
    ldg_apt = airports.copy()
    ldg_apt['Lat2'] = ldg_lat
    ldg_apt['Lon2'] = ldg_lon
    ldg_apt['Dist'] = ldg_apt.apply(get_dist, axis=1)

    # Extracting landing airport and runway characteristics
    # selecting correct landing airport
    if ldg_apt.loc[ldg_apt['Dist'].idxmin()]['Dist'] < 20:
        datapoint_dict['ldg_apt_id'] = ldg_apt.loc[ldg_apt['Dist'].idxmin()]['id']
        datapoint_dict['ldg_apt_ident'] = ldg_apt.loc[ldg_apt['Dist'].idxmin()]['ident']
        datapoint_dict['ldg_apt_elevation_ft'] = ldg_apt.loc[ldg_apt['Dist'].idxmin()]['elevation_ft']
    else:
        datapoint_dict['ldg_apt_id'], datapoint_dict['ldg_apt_ident'] = 'Unknown', 'Unknown'
        print('Unknown landing airport, skipping this file')
        not_compiled_list[f_list[i]] = 'Unknown Airport'
        continue

    ldg_rwy = runways.copy()
    ldg_rwy['Destination_id'] = datapoint_dict['ldg_apt_id']
    ldg_rwy['Destination_Ident'] = datapoint_dict['ldg_apt_ident']
    ldg_rwy['Heading_landing'] = datapoint_dict['Heading_landing'].round()

    # Selecting correct runway, making sure the runway sheet have landing airport data
    if (ldg_apt.loc[ldg_apt['Dist'].idxmin()]['ident'] == ldg_rwy['airport_ident']).any():
        ldg_rwy = ldg_rwy[ldg_rwy['airport_ident'] == ldg_rwy['Destination_Ident']]
    else:
        print('landing airport do not exist in runway sheet')
        not_compiled_list[f_list[i]] = 'Landing airport is not in runway data'
        continue

    ldg_rwy['Heading_delta'] = abs(ldg_rwy['Heading_landing'] - ldg_rwy['heading_degT'])

    # Making sure if heading is not NaN in runway sheet
    if np.isnan(ldg_rwy['heading_degT']).any():
        not_compiled_list[f_list[i]] = 'Heading Delta was NaN'
        print('Heading delta is None, skipping this file')
        continue
    else:
        # selecting the correct runway
        if (ldg_rwy['Heading_delta'] < 1.5).any():
            ldg_rwy = ldg_rwy[ldg_rwy['Heading_delta'] < 1.5]
        else:
            if (np.floor(ldg_rwy['Heading_delta'] / 10) * 10 == 360.0).any():
                ldg_rwy = ldg_rwy[np.floor(ldg_rwy['Heading_delta'] / 10) * 10 == 360.0]
            elif (np.ceil(ldg_rwy['Heading_delta'] / 10) * 10 == 360.0).any():
                ldg_rwy = ldg_rwy[np.ceil(ldg_rwy['Heading_delta'] / 10) * 10 == 360.0]

    ldg_rwy['Lat2'] = ldg_threshold_lat
    ldg_rwy['Lon2'] = ldg_threshold_lon
    ldg_rwy['Dist'] = ldg_rwy.apply(get_dist, axis=1)
    if ldg_rwy.shape[0] == 1:
        datapoint_dict['ldg_apt_rwy'] = ldg_rwy.iloc[0]['ident']
    else:
        datapoint_dict['ldg_apt_rwy'] = ldg_rwy.loc[ldg_rwy['Dist'].idxmin()]['ident']

    table = pd.DataFrame(datapoint_dict, index=[0])
    table1.loc[i] = table.loc[0]


# Exporting the table to CSV file
table1.to_csv(table_dir + 'table.csv', index=False)
unused_files = pd.Series(not_compiled_list)
unused_files.to_csv(table_dir + 'unused_file_list.csv', index=True)
