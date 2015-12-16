setwd('/home/ubuntu/coursera_machine_learning')

##########
# Data Wrangling and Covariate Creation
##########

library(caret)
library(ggplot2)

set.seed(666)

# Load training data and clean out "#DIV/0!" errors
train.start <- read.table('data/pml-training.csv',sep=',', header=T, stringsAsFactors=F)

x <- readLines('data/pml-training.csv')
y <- gsub( "\"#DIV/0!\"", "NA", x )
cat(y, file="data/clean.train.csv", sep="\n")

# Load cleaned training data and testing data
train.clean <- read.table('data/clean.train.csv',sep=',', header=T, stringsAsFactors=F)
test.quiz <- read.table('data/pml-testing.csv',sep=',', header=T, stringsAsFactors=F)

# identify the purely numeric covariates
nums <- sapply(train.clean, is.numeric)
train <- train.clean[,nums]
test.quiz <- test.quiz[,nums]

# remove index and timestamp columns
train <- train[, c(-1:-4)]
test.quiz <- test.quiz[, c(-1:-4)]

# remove the near-zero variables
nsv <- nearZeroVar(train,saveMetrics=TRUE)
train <- train[,!nsv$zeroVar]
test.quiz <- test.quiz[,!nsv$zeroVar]

# remove columns that are more than 50% NA
test.quiz <- test.quiz[ , apply(train, 2, function(x) (length(which(is.na(x)==F))/length(x)) > .5)]
train <- train[ , apply(train, 2, function(x) (length(which(is.na(x)==F))/length(x)) > .5)]

##########
# Data Slicing and Predictor Plotting
##########

# split data
train<-cbind(train, train.clean$classe)
names(train)[ncol(train)] <- 'classe'

train.partition <- createDataPartition(y=as.character(train.clean$classe), p=.75, list=F)

train.set <- train[train.partition,]
test.set <- train[-train.partition,]
train.classe <- train.set$classe
test.classe <- test.set$classe
train.set <- train.set[,-ncol(train.set)]
test.set <- test.set[,-ncol(test.set)]

##########
# Visualize
##########
train.qplot <- cbind(train.set, train.classe)

pairs(train.qplot[,c(sample(1:ncol(train.qplot[,-ncol(train.qplot)]), 5), ncol(train.qplot))], 
      main = "Randomized View of training data pairs", pch = 21, 
      bg = c("orange", "green3", "red", "purple", "blue")[unclass(train.qplot$train.classe)], 
      lower.panel=NULL, font.labels=2, cex.labels=1.0)

require(gridExtra)
plot1 <- qplot(yaw_arm, train.classe, data=train.qplot, colour=train.classe)
plot2 <- qplot(gyros_dumbbell_x, train.classe, data=train.qplot, colour=train.classe)
plot3 <- qplot(roll_dumbbell, magnet_forearm_y, data=train.qplot, colour=train.classe)
grid.arrange(plot1, plot2, plot3, ncol=2, nrow=2)

preProc <- preProcess(train.set, method=c("knnImpute"))
train.knn <- predict(preProc, train.set)
test.knn <- predict(preProc, test.set)
test.quiz.knn <- predict(preProc, test.quiz)
prePCA <- preProcess(train.knn, method="pca")
train.pca <- predict(prePCA, train.knn)
test.pca <- predict(prePCA, test.knn)
test.quiz.pca <- predict(prePCA, test.quiz.knn)

featurePlot(train.pca, train.classe)

##########
# Fit Models and Predict
##########

# Decision Tree
tc <- trainControl(method="cv", number=5)
modFit <- train(train.pca, train.classe, method="rpart", trControl=tc)
pred <- predict(modFit, test.pca)
mean(pred == test.classe)

# Linear Discriminant Analysis
modFit <- train(train.pca, train.classe, method="lda", trControl=tc)
pred <- predict(modFit, test.pca)
mean(pred == test.classe)

# Tree-Bagging
modFit <- train(train.pca, train.classe, method="treebag", trControl=tc)
pred <- predict(modFit, test.pca)
mean(pred == test.classe)

table(pred,test.classe)

suppressWarnings(modFit <- train(train.pca, train.classe, method="rf", trControl=tc, ntree=250))
pred <- predict(modFit, test.pca)
mean(pred == test.classe)




pred.quiz <- predict(modFit, test.quiz.pca)
pred.quiz


pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pred.quiz)
