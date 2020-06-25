# Real world data sets for testing HVG construction and merging algorithms

All data are provided in the `/data` directory of the repository.



## Financial Data

The financial time series data are tick-level foreign exchange data (bid price) covering the currencies and months in the following table. Timestamps have been removed leaving the raw sequence data in the CSV files with one value per row.

| File name     | Currencies | Month   | Length  |
| ------------- | ---------- | ------- | ------- |
| finance01.csv | AUD/USD    | 2019-05 | 1735110 |
| finance02.csv | USD/CAD    | 2019-06 | 1855446 |
| finance03.csv | EUR/CHF    | 2019-07 | 1681146 |
| finance04.csv | EUR/GBP    | 2019-08 | 2255368 |
| finance05.csv | AUD/NZD    | 2019-09 | 1646377 |

These time series were extracted from freely downloadable data sets provided by TrueFX [1].



## EEG Data

The brain data are taken from the American Epilepsy Society Seizure Prediction Challenge dataset [2]. In particular the selected time series are the first five channels in the data file 'Patient_1_preictal_segment_0001.mat' which records the first 10 minutes of brain activity in a window of roughly one hour prior to a seizure. Each selected channel is sampled at 5KHz giving $3\times 10^6$ measurement values per channel for the 10 minute recording.


| File name | Channel | Length  |
| --------- | ------- | ------- |
| eeg01.csv | 1       | 3000000 |
| eeg02.csv | 2       | 3000000 |
| eeg03.csv | 3       | 3000000 |
| eeg04.csv | 4       | 3000000 |
| eeg05.csv | 5       | 3000000 |



## References

[1] https://www.truefx.com/truefx-historical-downloads/

[2] https://www.kaggle.com/c/seizure-prediction/data