## Beginning of Code
## Set up Libraries here
library(tensorflow)
library(keras)
library(ggplot2)
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

## Scale Data (X)
X.sc <- scale(spam.dt[, -label.col, with=FALSE])
X.train.mat <- X.sc[is.train, ]
X.test.mat <- X.sc[is.test, ]
# Matrices to Arrays, Possibly not necessary
X.train.a <- array(X.train.mat, dim(X.train.mat))
X.test.a <- array(X.test.mat, dim(X.test.mat))

## Set up Y.train and Y.test
Y.train <- Y.Arr[is.train]
Y.test <- Y.Arr[is.test]

#Good To plug into Tensorflow

n.splits <- 10
split.metrics.list <- list()
for( split.i in 1:n.splits)
{
  ## Make Model here.
  model <- keras_model_sequential() %>% 
    layer_flatten(input_shape = ncol(X.train.mat)) %>% #Input layer
    layer_dense(units = 100, activation = "sigmoid", use_bias = FALSE) %>% # Hidden layer, change # of hidden units here? (10, 100, 1000)
    layer_dense(units = 1, activation = "sigmoid", use_bias = FALSE) # Output Layer
  
  ## Compile Model here.
  model %>% 
    compile(
      loss = "binary_crossentropy",
      optimizer = "sgd",
      metrics = "accuracy"
    )
  ## Fit model here
  result <- model %>% 
    fit(
      x = X.train.mat, y = Y.train,
      epochs = 100,
      validation_split = 0.4, #0.4 means 40% validation data
      verbose = 2
    )
  plot(result)
  metrics.wide <- do.call(data.table::data.table, result$metrics)
  metrics.wide[, epoch := 1:.N]
  split.metrics.list[[split.i]] <- data.table::data.table(
    split.i, metrics.wide)
}
split.metrics <- do.call(rbind, split.metrics.list)

split.means <- split.metrics[, .(
  mean.val.loss=mean(val_loss), 
  sd.val.loss=sd(val_loss)
  ), by=epoch]

min.dt <- split.means[which.min(mean.val.loss)]
min.dt[, point := "min"]
# ggplot2 lib called in function head
ggplot()+
  geom_ribbon(aes(
    x=epoch, ymin=mean.val.loss-sd.val.loss, ymax=mean.val.loss+sd.val.loss),
    alpha=0.5,
    data = split.means
  )+
  geom_point(aes(
    x=epoch, y=mean.val.loss),
    data = split.means)+
  geom_point(aes(
    x=epoch, y=mean.val.loss, color=point),
    data=min.dt
  )




