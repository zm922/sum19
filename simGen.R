Dat = read.csv(file="sp500_percent_data1.csv", header=TRUE, sep=",")
rownames(Dat) <- Dat$date
Dat <- Dat[-c(1:2)]
# n.stocks = dim(Dat)[2]
n.stocks = 100

Dat = Dat[, 1:n.stocks, drop = FALSE]
cnames = colnames(Dat)

# model spec

uspec.n = multispec(replicate(n.stocks, ugarchspec(mean.model = list(armaOrder = c(1,0)))))

multf = multifit(uspec.n, Dat)

spec2 = dccspec(uspec = uspec.n, dccOrder = c(1, 1), distribution = 'mvnorm')

fit2 = dccfit(spec2, data = Dat, fit.control = list(eval.se=FALSE), fit = multf)



vspec = vector(mode = "list", length = n.stocks)
midx = fit2@model$midx
mpars = fit2@model$mpars
for(i in 1:n.stocks){
  vspec[[i]] = uspec
  setfixed(vspec[[i]])<-as.list(mpars[midx[,i]==1, i])
}
dccfix = as.list(coef(fit2, "dcc"))
spec2 = dccspec(uspec = multispec( vspec ), 
                dccOrder = c(1,1),  distribution = "mvnorm",
                fixed.pars = dccfix)

presigma = tail(sigma(fit2), 1)
preresiduals = tail( residuals(fit2), 1)
prereturns = tail( as.matrix(Dat), 1 )
sim1 = dccsim(fitORspec = fit2, n.sim = 1000, n.start = 0, m.sim = 1, startMethod = "sample", 
              presigma = presigma, preresiduals = preresiduals, prereturns = prereturns, 
              preQ = last(rcor(fit2, type = "Q"))[,,1], Qbar = fit2@mfit$Qbar, 
              preZ = tail(fit2@mfit$stdresid, 1),
              rseed = c(100, 200), mexsimdata = NULL, vexsimdata = NULL)

