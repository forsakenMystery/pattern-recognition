savePlots = F

library(MASS)
library(nnet)

set.seed(10)

# Generate the data suggested (basically the XOR data set):
#
nTrain = 15
X      = runif( nTrain )
X      = sort(X)
X      = data.frame(X)
YTrain = 0.3 + 0.2 * cos( 2 * pi * X ) 

# Generate NTest new vectors and predict their response: 
#
XTest = seq(0.,1.,length=10)
XTest = sort( XTest )
XTest = data.frame( XTest ) 
YTest = 0.3 + 0.2 * cos( 2 * pi * XTest ) 

# Train a neural network on this data
#
regNet = nnet( X, YTrain, size=2, trace=FALSE ) # the example calling sequence
YPredict = predict( regNet, XTest ) # the predictions

if( savePlots ){
  postscript("../../WriteUp/Graphics/Chapter4/prob_17_data.eps", onefile=FALSE, horizontal=FALSE)
}

matplot( X, YTrain, xlab='x', ylab='y', type="b", pch=18, col="black" )
matplot( XTest, YPredict, type="b", pch=19, col="red", add=T ) 
matplot( XTest, YTest, type="b", pch=20, col="green", add=T ) # the truth 

legend( 0.5, 0.45, legend=c( "training data", "testing data", "truth" ), pch=c(18,19,20), col=c("black","red","green") )

if( savePlots ){
  dev.off()
}

