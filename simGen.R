# load libraries
library(rmgarch)
library(RcppCNPy)
library(rTensor)

# Read in the data from data_create.ipynb
filepath = "/Users/zachariemartin/Desktop/School/Projects/summer2019/2/sum19/data/zprocess_data.csv"
Dat = read.csv(file=filepath, header=TRUE, sep=",")
rownames(Dat) <- Dat$date
Dat <- Dat[-c(1)]
# n.stocks = dim(Dat)[2]
n.stocks = 100
Dat = Dat[,1:n.stocks]

# select columns corresponding to list
#col.num <- which(colnames(Dat) %in% name_data)

# subset according to n.stocks
# col.num <- col.num[c(1:n.stocks)]
# save names of columns
#stock_names <- colnames(Dat)[c(col.num)]

# get those columns
#Dat = Dat[, col.num, drop = FALSE]

# Well first use the multifit function so that we obtain an object which
# can be passed to the dccfit, thus elimimating the need to estimate
# multiple times the first stage

# assume the same volitlity model for each asset
# specifies an AR(1)-GARCH(1,1) model
uspec.n = multispec(replicate(n.stocks, ugarchspec(mean.model = list(armaOrder = c(1,0)))))

# estimate these univariate GARCH models
multf = multifit(uspec.n, Dat)

# normal dist - student also supported
# correlation specification
# In this specification we have to state how the univariate volatilities are modeled 
# (as per uspec.n) and how complex the dynamic structure of the correlation matrix is 
# (here we are using the most standard dccOrder = c(1, 1) specification).
spec2 = dccspec(uspec = uspec.n, dccOrder = c(1, 1), distribution = 'mvnorm')

# estimate the model using the dccfit function
# We want to estimate the model as specified in spec2, using the data in Dat. 
# Importantly fit = multf indicates that we ought to use the already estimated 
# univariate models as they were saved in multf
fit2 = dccfit(spec2, data = Dat, fit.control = list(eval.se=FALSE), fit = multf)


presigma = tail(sigma(fit2), 1)
preresiduals = tail( residuals(fit2), 1)
prereturns = tail( as.matrix(Dat), 1 )

# simulate returns

sim1 = dccsim(fitORspec = fit2, n.sim = 3000, n.start = 0, m.sim = 1, startMethod = "sample", 
              presigma = presigma, preresiduals = preresiduals, prereturns = prereturns, 
              preQ = last(rcor(fit2, type = "Q"))[,,1], Qbar = fit2@mfit$Qbar, 
              preZ = tail(fit2@mfit$stdresid, 1),
              rseed = c(100, 200), mexsimdata = NULL, vexsimdata = NULL)

returns = data.frame(sim1@msim$simX)
#cov_mats = array(sim1@msim$simH)
cov_mats = rcov(sim1)
stocks = colnames(Dat)
colnames(returns) = stocks

# Get the model based time varying covariance (arrays) matrices
cov1 = rcov(fit2)  # extracts the covariance matrix

# save simulated cov matrices in numpy format
npySave("mgarch_sim_3000_cov100.npy", cov_mats)

# save simulated cov matrices in csv format
# write.csv(cov_mats,'mgarch_sim_3000_cov100.csv')

# save returns
#returns_df <- data.frame(returns)
#colnames(returns_df) <- stock_names
write.csv(returns,'mgarch_sim_3000_z100.csv')

