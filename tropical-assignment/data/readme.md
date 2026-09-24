# Data

The Chicago taxi case study uses the **Chicago Taxi Rides 2016** dataset available through Kaggle:

[Chicago Taxi Rides 2016](https://www.kaggle.com/datasets/chicago/chicago-taxi-rides-2016)

The raw taxi data are not included in this repository because of their size. To reproduce the case study, download the dataset and place the required files in this directory as described below.

## Required files

The analysis uses the January 2016 taxi-trip data together with the column-remapping file:

```text
data/
└── taxis/
    ├── chicago_taxi_trips_2016_01.csv
    └── column_remapping.json