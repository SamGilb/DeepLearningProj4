## Spam Upload
if(!file.exists("spam.data"))
{
  download.file( 
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data", "spam.data")
}

spam.dt <- data.table::fread("spam.data")
## Need to convert Spam DT to an array
label.col <- ncol(spam.dt)
Y.Arr <- array( spam.dt[[label.col]], nrow(spam.dt) )

set.seed(1)
fold.vec <- sample(rep(1:5, l=nrow(spam.dt)))
test.fold <- 1
is.test <- fold.vec == test.fold
is.train <- !is.test

#Scale Data (X)
X.sc <- scale(spam.dt[, -label.col, with=FALSE])
X.train.mat <- X.sc[is.train, ]
X.test.mat <- X.sc[is.test, ]
#Change Matrices to Arrays, Possibly not necessary
X.train.a <- array(X.train.mat, dim(X.train.mat))
X.test.a <- array(X.test.mat, dim(X.test.mat))

#Set up Y.train and Y.test
Y.train <- Y.Arr[is.train]
Y.test <- Y.Arr[is.test]

#Good To plug into Tensorflow