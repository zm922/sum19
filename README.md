



## Steps to train rl models

1. use data_create.ipynb notebook to create Z process data from price/returns data
2. use data_create.ipynb notebook to test Z process data for stationarity - save as .csv if satisfied
3. use simGen.R to train MGARCH model on Z process data created above
4. Export mgarch model parameters as .csv
5. if you'd like to see how data is simulated, check out generate_date.ipynb notebook, this function lives inside the rl algorithm code but I've extracted it to make explicit this process - in case we want to further examine the data being fed in to the rl algorithm 
6. run algorithm using drl/src/stock_trading.py (not completed yet); or if you want to prototype in notebook, use drl/src/prototyping.ipynb

## further work

- convergence of RL algorithm
  - data issues: is simulated data faithful to real data distribution?
  - simulating the spreads in python doesn't seem to be working
  - algorithm issues: choice of hyperparams, model types, e.g. lstm, cnn, etc.
- dcc model (in R) runs in to convergence problems for >100 assets
- 