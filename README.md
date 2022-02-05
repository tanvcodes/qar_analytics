# QAR Analytics Description

This repository contains scripts for woking with publicly available Quick Access Recorder (QAR) data rom a fleet of 35 BAE-146 aircraft. This project may be of particular interest for anyone wanting to get into flight data analysis. Additional details on the dataset are available on NASA's Dashlink Website (https://c3/nasa.gov/dashlink/projects85/)

## Quick Start (matlab_to_table.py)

Ensure that you have Python installed along with the latest version of Pandas.
This script will automatically download all fiiles associated wtih specified TailNo. The raw files are in matlab file format which are not ideal for analysis purposes. This script converts the matlab list format into tabular format with rows representing 1s time increments and the columns representing the different sensor parameters captured by the aircraft.

One output file will be generated for each flight and saved to the output directory specified by output_dir parameter. The output file format can be specified by the output_format parameter. Valid formats are 'parquet' or 'csv'

Thematlab_to_table.py script will also save meta data ( departure/arrival ariports, takeoff/arrival time, etc) associated with each flight to the output_dir directory.


## NASA_Decoder_Parquet.csv

Containts a list of sensor parameters along with the data type to use for the parquet file.


## Tailcounts.csv

Contains a list of all the tailnumbers in the enire data set. Also, contains the number of files er tail.


## USA_airport_Codes.csv

Contains a list of many airports in the US. Used to identify departure and destination airports.

## 666200402020631.mat

A sample file which is used to provide information on the columns/data types contained in the matlab file.


## airports.csv

Contains a list of US airport with the their location and elevation details. Used to extract the departure and destination airport location and elevation.


## runways.csv

Contains a list of airports with their runways and for each runway their heading angle, elevation and identificatin. Used to extract the landing runway angle, elevation.


## datapoint_extraction.py

This script will extract useful data from each csv file generated from matlab_to_table script at around landing and touchdown time.


## ML_Model.py

This script creates a machine learning model to predict whether its a hard landing or not.
