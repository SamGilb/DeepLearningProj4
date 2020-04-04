
#Code setup
library(tensorflow)
library(keras)
library(ggplot2)
library(data.table)
set.seed(1)


###############################################################################
###############################################################################
###################### DATA UPLOAD AND ORGANIZATION ###########################
###############################################################################
###############################################################################

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

###############################################################################
###############################################################################
############################# Train Model 1 ###################################
###############################################################################
###############################################################################

#Initialize model with architecture of (ncol(X), 10, 1)
model1 <- keras_model_sequential() %>% 
  layer_flatten(input_shape = ncol(X.train.mat)) %>% #Input layer
  layer_dense(units = 10, activation = "sigmoid", use_bias = FALSE) %>% # Hidden layer, change # of hidden units here? (10, 100, 1000)
  layer_dense(units = 1, activation = "sigmoid", use_bias = FALSE) # Output Layer

## Compile Model for binary classification
model1 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "sgd",
    metrics = "accuracy"
  )

## Fit model here
result1 <- model1 %>% 
  fit(
    x = X.train.mat, y = Y.train,
    epochs = 250,
    validation_split = 0.4, #0.4 means 40% validation data
    verbose = 2
  )

#store metrics for later use in plot section
metrics1 <- do.call(data.table, result1$metrics)
metrics1[, epoch := 1:.N]
min.metrics1 <- metrics1[which.min(val_loss)]
best_epochs1 <- min.metrics1$epoch

###############################################################################
###############################################################################
############################# Train Model 2 ###################################
###############################################################################
###############################################################################

#Initialize model with architecture of (ncol(X), 10, 1)
model2 <- keras_model_sequential() %>% 
  layer_flatten(input_shape = ncol(X.train.mat)) %>% #Input layer
  layer_dense(units = 100, activation = "sigmoid", use_bias = FALSE) %>% # Hidden layer, change # of hidden units here? (10, 100, 1000)
  layer_dense(units = 1, activation = "sigmoid", use_bias = FALSE) # Output Layer

## Compile Model for binary classification
model2 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "sgd",
    metrics = "accuracy"
  )

## Fit model here
result2 <- model2 %>% 
  fit(
    x = X.train.mat, y = Y.train,
    epochs = 250,
    validation_split = 0.4, #0.4 means 40% validation data
    verbose = 2
  )

#store metrics for later use in plot section
metrics2 <- do.call(data.table, result2$metrics)
metrics2[, epoch := 1:.N]
min.metrics2 <- metrics2[which.min(val_loss)]
best_epochs2 <- min.metrics2$epoch

###############################################################################
###############################################################################
############################# Train Model 3 ###################################
###############################################################################
###############################################################################

#Initialize model with architecture of (ncol(X), 10, 1)
model3 <- keras_model_sequential() %>% 
  layer_flatten(input_shape = ncol(X.train.mat)) %>% #Input layer
  layer_dense(units = 1000, activation = "sigmoid", use_bias = FALSE) %>% # Hidden layer, change # of hidden units here? (10, 100, 1000)
  layer_dense(units = 1, activation = "sigmoid", use_bias = FALSE) # Output Layer

## Compile Model for binary classification
model3 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "sgd",
    metrics = "accuracy"
  )

## Fit model here
result3 <- model3 %>% 
  fit(
    x = X.train.mat, y = Y.train,
    epochs = 250,
    validation_split = 0.4, #0.4 means 40% validation data
    verbose = 2
  )

#store metrics for later use in plot section
metrics3 <- do.call(data.table, result3$metrics)
metrics3[, epoch := 1:.N]
min.metrics3 <- metrics3[which.min(val_loss)]
best_epochs3 <- min.metrics3$epoch

###############################################################################
###############################################################################
################################# Plotting ####################################
###############################################################################
###############################################################################


ggplot()+
  
  #plot first neural network
  geom_line(aes(
    x=epoch, y=loss, color = '10 Hidden Units', linetype = 'train'),
    data = metrics1)+
  geom_line(aes(
    x=epoch, y=val_loss, color = '10 Hidden Units', linetype = 'validation'),
    data = metrics1)+
  geom_point(aes(
    x=epoch, y=val_loss, color='min'),
    data=min.metrics1
  )+
  
  #plot second neural network
  geom_line(aes(
    x=epoch, y=loss, color = '100 Hidden Units', linetype = 'train'),
    data = metrics2)+
  geom_line(aes(
    x=epoch, y=val_loss, color = '100 Hidden Units', linetype = 'validation'),
    data = metrics2)+
  geom_point(aes(
    x=epoch, y=val_loss, color='min'),
    data=min.metrics2
  )+
  
  #plot third neural network
  geom_line(aes(
    x=epoch, y=loss, color = '1000 Hidden Units', linetype = 'train'),
    data = metrics3)+
  geom_line(aes(
    x=epoch, y=val_loss, color = '1000 Hidden Units', linetype = 'validation'),
    data = metrics3)+
  geom_point(aes(
    x=epoch, y=val_loss, color='min'),
    data=min.metrics3
  )+
  labs(
    color = "Model",
    linetype = "set"
    
  )

###############################################################################
###############################################################################
########################### Re-train Model 1 ##################################
###############################################################################
###############################################################################

#Initialize model with architecture of (ncol(X), 10, 1)
model1 <- keras_model_sequential() %>% 
  layer_flatten(input_shape = ncol(X.train.mat)) %>% #Input layer
  layer_dense(units = 10, activation = "sigmoid", use_bias = FALSE) %>% # Hidden layer, change # of hidden units here? (10, 100, 1000)
  layer_dense(units = 1, activation = "sigmoid", use_bias = FALSE) # Output Layer

## Compile Model for binary classification
model1 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "sgd",
    metrics = "accuracy"
  )

## Fit model here
result1 <- model1 %>% 
  fit(
    x = X.train.mat, y = Y.train,
    epochs = best_epochs1,
    validation_split = 0, #train on whole training set
    verbose = 2
  )


###############################################################################
###############################################################################
############################# Re-train Model 2 ##################################
###############################################################################
###############################################################################

#Initialize model with architecture of (ncol(X), 10, 1)
model2 <- keras_model_sequential() %>% 
  layer_flatten(input_shape = ncol(X.train.mat)) %>% #Input layer
  layer_dense(units = 100, activation = "sigmoid", use_bias = FALSE) %>% # Hidden layer, change # of hidden units here? (10, 100, 1000)
  layer_dense(units = 1, activation = "sigmoid", use_bias = FALSE) # Output Layer

## Compile Model for binary classification
model2 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "sgd",
    metrics = "accuracy"
  )

## Fit model here
result2 <- model2 %>% 
  fit(
    x = X.train.mat, y = Y.train,
    epochs = best_epochs2,
    validation_split = 0, #train on whole training set
    verbose = 2
  )

###############################################################################
###############################################################################
########################### Re-train Model 3 ##################################
###############################################################################
###############################################################################

#Initialize model with architecture of (ncol(X), 10, 1)
model3 <- keras_model_sequential() %>% 
  layer_flatten(input_shape = ncol(X.train.mat)) %>% #Input layer
  layer_dense(units = 1000, activation = "sigmoid", use_bias = FALSE) %>% # Hidden layer, change # of hidden units here? (10, 100, 1000)
  layer_dense(units = 1, activation = "sigmoid", use_bias = FALSE) # Output Layer

## Compile Model for binary classification
model3 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "sgd",
    metrics = "accuracy"
  )

## Fit model here
result3 <- model3 %>% 
  fit(
    x = X.train.mat, y = Y.train,
    epochs = best_epochs3,
    validation_split = 0, #train on whole training set
    verbose = 2
  )


###############################################################################
###############################################################################
########################### Model Evaluation ##################################
###############################################################################
###############################################################################

#Evaluate first model
print("Model 1's loss and accuracy is:")
model1 %>%
  evaluate( X.test.mat, Y.test, verbose = 0)

#Evaluate second model
print("Model 2's loss and accuracy is:")
model2 %>%
  evaluate( X.test.mat, Y.test, verbose = 0)

#Evaluate third model
print("Model 3's loss and accuracy is:")
model3 %>%
  evaluate( X.test.mat, Y.test, verbose = 0)

#Evaluate baseline accuracy
print("Baseline Accuracy is:")
baseline.accuracy <- max(sum(Y.test == 1), sum(Y.test == 0)) / length(Y.test)
print(baseline.accuracy)
