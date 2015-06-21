# TODO: Add comment
# 
# Author: evberghe
###############################################################################

# prepare for parallel processing
library(doMC)
registerDoMC(cores=8)

# read data
setwd("/home/evberghe/workspace/datascience/ml/project")
datadir <- "./data"
testing <- read.table(paste(datadir, "pml-testing.csv", sep="/"), sep=",", header=TRUE, na.strings=c("", "#DIV/0!", "NA"))
training <- read.table(paste(datadir, "pml-training.csv", sep="/"), sep=",", header=TRUE, na.strings=c("", "#DIV/0!", "NA"))

library("caret")

trn <- training[, -(1:7)]
completeness <- apply(trn, 2, function(x) mean(!is.na(x)))
trn <- trn[, completeness>0.5]
tst <- testing[, -(1:7)]
tst <- tst[, completeness>0.5]

trtfrc <- 0.025
trtidx <- sample(c(TRUE, FALSE), size=nrow(trn), 
		replace=TRUE, prob=c(trtfrc, 1-trtfrc))
trt <- trn[trtidx, ]; trv <- trn[-trtidx, ]
nodesizes <- 1:10
nodenumbers <- 2^(1:10)
accurs <- matrix(rep(NA, length(nodesizes)*length(nodenumbers)), 
		ncol=length(nodenumbers))
for(i in 1:length(nodesizes)){
	for(j in 1:length(nodenumbers)){
		fit <- train(classe~., data=trt, method="rf", 
				nodesize=nodesizes[i], maxnodes=nodenumbers[j], ntree=500)
		accurs[i, j] <- mean(predict(fit, newdata=trv)==trv$classe)
		cat(sprintf("node size: %d; number of nodes %d; out of bag %f \n", nodesizes[i], nodenumbers[j], accurs[i, j]))
	}	
}

png("accurs.png")
image(accurs, axes=FALSE)
axis(1, at=(0:9)/9, labels=1:10)
axis(2, at=(0:9)/9, labels=1:10)

fit <- train(classe~., data=trn, method="rf", ntree=500)

# destroy the cluster when we're ready with it
stopCluster(cl)