# This script will download flight data from NASA's Dashlink (https://c3.nasa.gov/dashlink/projects/85/) website
# and then convert the file into tabular format (either parquet or csv) for analysis purposes.

# loading libraries

import pandas as pd
import numpy as np
import os
import wget
import sys
import zipfile
from math import sin, cos, sqrt, atan2, radians
from scipy.io import loadmat
import time

# Specify user configuration
TailNo = '652'  # See tailcounts.csv for list of aircraft numbers. Will only run one at at time
homeDir = 'C:\\Users\\kt733e\\Documents\\work\\Proj\\'  # Specify directory containing working files
output_format = 'parquet'   # File format for the output. Options are 'parquet' and 'csv'

# Assigning directories     
airport_dir = homeDir + 'USA_Airport_Codes.csv'  # Location of a list of airports

if output_format == "parquet":
    output_dir = homeDir + 'Tail' + TailNo + '/Parquet/'
elif output_format == 'csv':
    output_dir = homeDir + 'Tail' + TailNo + '/csv/'
else:
    exit()


# Function to upload files to google cloud storage
'''
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    '''


# Function to calculate individual sensor states
# This is needed because many binary sensors have been combined into a single column.
# For example, each of the engines have their own anti-ice system (EAI). This system
# can be either 'On' or 'Off' for a given engine. However, rather than reporting as
# 4 separate columns (Engine 1 ON/OFF, Engine 2 ON/OFF) NASA has used a single column
# to represent the unique number of switch possibilities.
def int_to_bin(x, n):
    n = n + 1
    try:
        x = int(bin(x)[n])

    except:
        x = 0
    return x


# Function to calculate the distance between two points in km
def get_dist(row):
    # Radius of the earth
    R = 6373.0
    
    # Converting to radians
    lat1 = radians(row['Latitude'])
    lon1 = radians(row['Longitude'])
    lat2 = radians(row['Lat2'])
    lon2 = radians(row['Lon2'])

    # Calculating delta between lat and long
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Calculating the distance between two points
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


# ============================START OF PROGRAM=========================================
# Getting list of airports
airports = pd.read_csv(airport_dir)

# Getting number of files to download
counts = pd.read_csv(homeDir + 'tailcounts.csv')
counts['Tail'] = counts['Tail'].apply(str)
reps = counts[counts['Tail'] == TailNo].Count[0]
reps = reps + 1

# Getting list of column information
alias_csv = homeDir + 'NASA_Decoder_Parquet.csv'
alias = pd.read_csv(alias_csv)

# Defining location to save sensor information
col_dir = homeDir + 'Tail' + TailNo

# Creating new folder to save files, of it doesn't already exist
try:
    os.mkdir(col_dir)
    os.mkdir(output_dir)
except:
    pass

# Getting meta information for each sensor based on a single sample file
flight = loadmat(homeDir + '666200402020631.mat', squeeze_me=True)
sensorName = []
for y in flight:
    sensorName.append(y)

sensorName.sort()

# Removing non-sensor values from list of sensor names
nonSensors = ['ACMT', 'VAR_1107', 'VAR_2670', 'VAR_5107', 'VAR_6670', '__globals__', '__header__', '__version__',
              'GMT_HOUR', 'GMT_MINUTE', 'GMT_SEC', 'DATE_DAY', 'DATE_MONTH', 'DATE_YEAR', 'ACID', 'DVER_1', 'DVER_2']
sensorName = [x for x in sensorName if x not in nonSensors]

# Grabbing the update frequencies
sensorFreq = []
sensorIndex = []
for i, x in enumerate(sensorName):
    try:
        freq = flight[sensorName[i]].take(0)[1]
        sensorFreq.append(freq)
        sensorIndex.append(sensorName[i])
    except:
        print("skipped " + sensorName[i])

sensorFreq = pd.Series(sensorFreq, index=sensorIndex)


'''
for matNo in range(1, reps):
    matZip = homeDir + 'Tail' + TailNo + '_' + str(matNo) + '.zip'
    matDir = homeDir + 'Tail' + TailNo + '/matlab'

    # Unzipping matlab files
    zipRef = zipfile.ZipFile(matZip, 'r')
    zipRef.extractall(matDir)
    zipRef.close()
    # os.removed(matZip)

    # getting list of files
    f_list = os.listdir(matDir)
    f_list.sort()
    n_files = len(f_list)

    # looping through flight data
    '''
matDir = homeDir + 'Tail' + TailNo + '\\matlab'
f_list = os.listdir(matDir)
f_list.sort()
n_files = len(f_list)

# Loop will execute for the number of times specified in the tailcounts 'Count' column
# Effectively this loop downloads a ziped file from NASA, unzips the file, then 
# loops through each file and converts it from a matlab file (list of lists) into
# a tabular format that can be saved as a parquet file.
for matNo in range(1, reps):
    
    # Downloading flight files from NASA
    url = 'https://c3.nasa.gov/dashlink/static/media/dataset/Tail_' + TailNo + '_' + str(matNo) + '.zip'
    matZip = homeDir + 'Tail' + TailNo + '_' + str(matNo) + '.zip'
    matDir = homeDir + 'Tail' + TailNo + '/matlab'
    wget.download(url, matZip)

    # Unzipping matlab files
    zipRef = zipfile.ZipFile(matZip, 'r')
    zipRef.extractall(matDir)
    zipRef.close()
    os.remove(matZip)
    
    # Getting list of files contained in the zip
    f_list = os.listdir(matDir)
    f_list.sort()
    n_files = len(f_list)
    
    # Looping through the unzipped matlab files
    for j, f_name in enumerate(f_list[3594:]):
        # Getting time (to compute the processing time)
        s_time = time.time()

        # loading flight data
        try:
            flight = loadmat(matDir + '\\' + f_name, squeeze_me=True)

            # grabbing meta data for the flight of interest
            flightNo = f_name[:15]
            tail_no = f_name[:3]
            f_date = f_name[3:7] + "-" + f_name[7:9] + "-" + f_name[9:11] + " " + f_name[11:13] + ":" + f_name[
                                                                                                        13:15] + ":00"
            # Appending of an 'A' to the flight number of there are multiple of the same flight number
            if len(f_name) > 19:
                flightNo = flightNo + 'A'

            # Extracting sensor information
            sensor = pd.DataFrame(columns=['UTC', 'item', 'value'])

            for i, x in enumerate(sensorName):
                freqX = str(((1 / sensorFreq[i]) * 1000)) + 'ms'
                sensorX = flight[x].take(0)
                sensorX = pd.DataFrame(sensorX[0], columns=['value'])
                sensorX['item'] = x
                sensorX['UTC'] = pd.date_range(f_date, periods=len(sensorX.index), freq=freqX)
                sensorX = sensorX[['UTC', 'item', 'value']]
                sensor = sensor.append(sensorX)
                
            # sorting by timestamp
            sensor = sensor.sort_values(by=['UTC', 'item'])

            # Reshaping into wide format
            sensor = sensor.pivot_table(index='UTC', values='value', columns='item')
            sensor = sensor.reset_index()

            # Forward filling blanks
            sensor = sensor.ffill()

            # Downsampling to 1Hz
            # Take the max of the accelerometer readings
            # Otherwise take the first reading
            sensor['UTC'] = sensor['UTC'].astype('datetime64[s]')
            cols = sensor.columns
            accel_list = ['BLAC', 'CTAC', 'FPAC', 'LATG', 'LONG', 'VRTG']
            agg_type = ['max' if x in accel_list else 'first' for x in cols]
            agg_dict = dict(zip(cols, agg_type))
            sensor = sensor.groupby('UTC').agg(agg_dict)
            sensor = sensor.reset_index(drop=True)

            # Determining bleed air valve positions
            sensor["BLV1"] = sensor["BLV"].apply(lambda x: int_to_bin(int(x), 1))
            sensor["BLV2"] = sensor["BLV"].apply(lambda x: int_to_bin(int(x), 2))
            sensor["BLV3"] = sensor["BLV"].apply(lambda x: int_to_bin(int(x), 3))
            sensor["BLV4"] = sensor["BLV"].apply(lambda x: int_to_bin(int(x), 4))
            sensor.drop('BLV', axis=1, inplace=True)

            # Determining DFGS Master status
            sensor["DFGS1"] = sensor["DFGS"].apply(lambda x: int_to_bin(int(x), 1))
            sensor["DFGS2"] = sensor["DFGS"].apply(lambda x: int_to_bin(int(x), 2))
            sensor.drop('DFGS', axis=1, inplace=True)

            # Determining engine antiice status
            sensor["EAI1"] = sensor["EAI"].apply(lambda x: int_to_bin(int(x), 1))
            sensor["EAI2"] = sensor["EAI"].apply(lambda x: int_to_bin(int(x), 2))
            sensor["EAI3"] = sensor["EAI"].apply(lambda x: int_to_bin(int(x), 3))
            sensor["EAI4"] = sensor["EAI"].apply(lambda x: int_to_bin(int(x), 4))
            sensor.drop('EAI', axis=1, inplace=True)

            # Determining whether FADEC had failed on an engine
            sensor["FADF1"] = sensor["FADF"].apply(lambda x: int_to_bin(int(x), 1))
            sensor["FADF2"] = sensor["FADF"].apply(lambda x: int_to_bin(int(x), 2))
            sensor["FADF3"] = sensor["FADF"].apply(lambda x: int_to_bin(int(x), 3))
            sensor["FADF4"] = sensor["FADF"].apply(lambda x: int_to_bin(int(x), 4))
            sensor.drop('FADF', axis=1, inplace=True)

            # Determining the FADEC status per engine
            sensor["FADS1"] = sensor["FADS"].apply(lambda x: int_to_bin(int(x), 1))
            sensor["FADS2"] = sensor["FADS"].apply(lambda x: int_to_bin(int(x), 2))
            sensor["FADS3"] = sensor["FADS"].apply(lambda x: int_to_bin(int(x), 3))
            sensor["FADS4"] = sensor["FADS"].apply(lambda x: int_to_bin(int(x), 4))
            sensor.drop('FADS', axis=1, inplace=True)

            # Determining whether the aircraft is over a marker
            sensor["MRK1"] = sensor["MRK"].apply(lambda x: int_to_bin(int(x), 1))
            sensor["MRK2"] = sensor["MRK"].apply(lambda x: int_to_bin(int(x), 2))
            sensor["MRK3"] = sensor["MRK"].apply(lambda x: int_to_bin(int(x), 3))
            sensor.drop('MRK', axis=1, inplace=True)

            # Determining whether the oil pressure is low
            sensor["OIPL1"] = sensor["OIPL"].apply(lambda x: int_to_bin(int(x), 1))
            sensor["OIPL2"] = sensor["OIPL"].apply(lambda x: int_to_bin(int(x), 2))
            sensor["OIPL3"] = sensor["OIPL"].apply(lambda x: int_to_bin(int(x), 3))
            sensor["OIPL4"] = sensor["OIPL"].apply(lambda x: int_to_bin(int(x), 4))
            sensor.drop('OIPL', axis=1, inplace=True)

            # Determine if the air conditioning packs are operating
            sensor["PACK1"] = sensor["PACK"].apply(lambda x: int_to_bin(int(x), 1))
            sensor["PACK2"] = sensor["PACK"].apply(lambda x: int_to_bin(int(x), 2))
            sensor.drop('PACK', axis=1, inplace=True)

            # Determining whether there is a pylon overheat event
            sensor["POVT1"] = sensor["POVT"].apply(lambda x: int_to_bin(int(x), 1))
            sensor["POVT2"] = sensor["POVT"].apply(lambda x: int_to_bin(int(x), 2))
            sensor["POVT3"] = sensor["POVT"].apply(lambda x: int_to_bin(int(x), 3))
            sensor["POVT4"] = sensor["POVT"].apply(lambda x: int_to_bin(int(x), 4))
            sensor.drop('POVT', axis=1, inplace=True)

            # converting columns types
            '''
            for x, col in enumerate(sensor.columns):
                # print(col, " ", sensor[col].dtype, " ", alias[alias['Parameter']==col].Dtype.any())
                if (alias[alias['Parameter'] == col].Dtype.any() != 'NoChange'):
                    sensor[col] = sensor[col].astype(alias[alias['Parameter'] == col].Dtype.any())'''
            
            # Inverting weight on wheels logic so that 1 =  on ground , 0= in air
            sensor["WOW"] = sensor["WOW"].replace({False: True, True: False})

            # Adding ID
            sensor["FLTID"] = f_name[:-4]
            sensor["FLTID"] = sensor["FLTID"].astype('str')

            # Adding wind speed
            sensor["wind_delta"] = sensor['TH'] - sensor['WD']
            sensor["HEADWIND"] = sensor["wind_delta"].apply(lambda x: cos(x * 3.14 / 180))
            sensor["HEADWIND"] = sensor["HEADWIND"] * sensor["WS"]
            sensor["XWIND"] = sensor["wind_delta"].apply(lambda x: sin(x * 3.14 / 180))
            sensor["XWIND"] = sensor["XWIND"] * sensor["WS"]
            sensor.loc[sensor['WOW'] == False, 'HEADWIND'] = 0
            sensor.loc[sensor['WOW'] == False, 'XWIND'] = 0
            sensor = sensor.drop(columns='wind_delta')
            sensor["HEADWIND"] = sensor["HEADWIND"].astype('float32')
            sensor["XWIND"] = sensor["XWIND"].astype('float32')

            # Getting takeoff and landing information
            to_idx = sensor[sensor["WOW"] == False].first_valid_index()
            ldg_idx = sensor[sensor["WOW"] == False].last_valid_index()

            # Creating meta data if flight departed
            if to_idx is None:
                # Filling in NAs for maintenance and ground only operations
                print("Not a flight. Skipping upload")

            else:
                # Gathering position and time data for flights
                to_lat = sensor.iloc[to_idx]['LATP']
                to_lon = sensor.iloc[to_idx]['LONP']
                ldg_lat = sensor.iloc[ldg_idx]['LATP']
                ldg_lon = sensor.iloc[ldg_idx]['LONP']
                to_time = pd.Series(sensor.iloc[to_idx]['UTC']).dt.round('S')[0]
                ldg_time = pd.Series(sensor.iloc[ldg_idx]['UTC']).dt.round('S')[0]
                flt_time = round((ldg_time - to_time) / np.timedelta64(1, 'h'), 1)

                # Adding elapsed flight time
                sensor['ELAPSED_TIME'] = round((sensor.UTC - to_time) / np.timedelta64(1, 'm'), 1)
                sensor['ELAPSED_TIME'] = sensor['ELAPSED_TIME'].astype('float32')

                # Calculating distance remaining
                sensor['DIST_REMAIN'] = sensor['GS'].cumsum()
                sensor['DIST_REMAIN'] = sensor['DIST_REMAIN'] * 0.5144 * (1 / 16) / 1000
                ldg_dist = sensor.iloc[ldg_idx]['DIST_REMAIN']
                sensor['DIST_REMAIN'] = ldg_dist - sensor['DIST_REMAIN']
                sensor['DIST_REMAIN'] = sensor['DIST_REMAIN'].astype('float32')

                # Getting takeoff airport
                to_apt = airports
                to_apt['Lat2'] = to_lat
                to_apt['Lon2'] = to_lon
                to_apt['Dist'] = to_apt.apply(get_dist, axis=1)
                if to_apt.loc[to_apt['Dist'].idxmin()]['Dist'] < 20:
                    to_apt = to_apt.loc[to_apt['Dist'].idxmin()]['locationID']
                else:
                    to_apt = 'Unknown'

                # Getting destination airport
                ldg_apt = airports
                ldg_apt['Lat2'] = ldg_lat
                ldg_apt['Lon2'] = ldg_lon
                ldg_apt['Dist'] = ldg_apt.apply(get_dist, axis=1)
                if ldg_apt.loc[ldg_apt['Dist'].idxmin()]['Dist'] < 20:
                    ldg_apt = ldg_apt.loc[ldg_apt['Dist'].idxmin()]['locationID']
                else:
                    ldg_apt = 'Unknown'
                # Tracking flight info
            meta = pd.DataFrame(
                {'flightNo': str(f_name[:-4]), 'origin': str(to_apt), 'destination': str(ldg_apt),
                 'takeoff_time': to_time, 'landing_time': ldg_time, 'flight_time': float(flt_time)}, index=[0])

            # Saving data to big query
            # sensor.to_gbq('qar.tail' + str(TailNo), if_exists='append')
            # meta.to_gbq('qar.meta'+ str(TailNo), if_exists='append')
            if output_format == 'parquet':
                parquet_file = f_name[:-4] + '.parquet.gz'
                sensor.to_parquet(output_dir + parquet_file, compression='gzip', index=False)
            elif output_format == 'csv':
                csv_file = f_name[:-4] + '.csv'
                sensor.to_csv(output_dir + csv_file, index=False)
            else:
                print('Invalid output file format. output_format must be either csv or parquet')
                exit()

            # Saving data to parquet, then uploading to google storage
            # upload_blob("bae146", CSV_dir + parquet_file, 'sensors/tail' + tail_no + '/' + parquet_file)
            # os.remove(CSV_dir + parquet_file)

            # Saving meta data
            # upload_blob("bae146", CSV_dir + meta_file, 'meta/tail' + tail_no + '/' + meta_file)
            # os.remove(parquet_dir + meta_file)

        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Unexpected error:", exc_type, " Line Number:", exc_tb.tb_lineno)
            print('Error on file: ' + f_name + '. No data written')
            with open(homeDir + 'err_log.txt', 'a') as err_log:
                err_log.write(f_name + '\n')

        # Removing matlab file
        try:
            os.remove(matDir + '/' + f_name)            
        except:
            print("Could not remove file: " + f_name)
            
        # Printing status
        loop_time = time.time() - s_time
        print('Tail No: ' + TailNo + '. MatLab File: ' + str(matNo) + '/' + str(reps) + '. Flight No: ' + str(j) + '/'
              + str(n_files) + '. Timer: ' + str(loop_time) + "s")

        