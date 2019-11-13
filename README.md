1. use data_create.ipynb to create Z process data from stock returns
2. use data_create.ipynb to test Z process data for stationarity
3. use simGen.R to simulate MGARCH data from Z process data created above
4. use data_transform.ipynb to transform data into format used by algorithm
5. view simulated data using data_view.ipynb
6. write simulated data to h5 file using sum19/drl/src/utils/write_to_h5.ipynb
7. run algorithm using stock_trading.py or if you want to prototype in notebook, use ddpg100_experiments.ipynb
