library(dplyr)
library(keras)
library(ggplot2)
library(ggpointdensity)
library(tidyr)
library(patchwork)
cqdata2 -> chongqingall

chongqingall <- chongqingall %>%
  mutate(across(c(1:3,5), ~lag(., 1), .names = "{col}_lag1"),   # lag=1
         across(c(1:3,5), ~lag(., 2), .names = "{col}_lag2"),   # lag=2
         across(c(1:3,5), ~lag(., 3), .names = "{col}_lag3"))



###################### TRANMODEL
for (i in 1:n_samples) {
  for (t in 1:3) {  
    
    lag_col <- paste0("lag", t)
    
   
    X_processed[i, t, 1] <- X_train[i, paste0("RES_", lag_col)]
    X_processed[i, t, 1] <- X_train[i, paste0("rhu_avg_", lag_col)]
    X_processed[i, t, 2] <- X_train[i, paste0("tem_avg_", lag_col)]
    X_processed[i, t, 3] <- X_train[i, paste0("wha2_", lag_col)]
  }
}


X_processed <- as.array(X_processed)

############  TESTMODEL
n_samples <- nrow(X_test)
X_test_processed <- array(0, dim = c(n_samples, 3, 3)) 


for (i in 1:n_samples) {
  for (t in 1:3) {
    
    lag_col <- paste0("lag", t)
    
   
    #X_test_processed[i, t, 1] <- X_test[i, paste0("RES_", lag_col)]
    X_test_processed[i, t, 1] <- X_test[i, paste0("rhu_avg_", lag_col)]
    X_test_processed[i, t, 2] <- X_test[i, paste0("tem_avg_", lag_col)]
    X_test_processed[i, t, 3] <- X_test[i, paste0("wha2_", lag_col)]
  }
}

X_test_processed <- as.array(X_test_processed) 



y_train <- y[train_indices]
y_train <- matrix(y_train, ncol = 1)
y_test <- y[-train_indices]
y_test <- matrix(y_test, ncol = 1)
dim(y_train)
dim(y_test)
###################  RUNNING

tensorflow::tf$keras$backend$clear_session()

input_time_series <- layer_input(shape = c(3, 3))   ##shape = c(3, 4)


lstm_layer <- input_time_series %>%
  layer_lstm(units = 64, dropout = 0.1, recurrent_dropout = 0.1,return_sequences = TRUE) %>%
  layer_lstm(units = 32, return_sequences = FALSE) %>% 
  layer_dense(units = 16, activation = "relu")


#output <- lstm_layer %>% layer_dense(units = 1, activation = "linear") 
output <- lstm_layer %>% layer_dense(units = 1)


model_lstm <- keras_model(inputs = input_time_series, outputs = output)


model_lstm %>% compile(loss = 'mean_squared_error', optimizer = 'adam')

lr_scheduler <- callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  factor = 0.5,  
  patience = 3,
  min_lr = 1e-6 
)

early_stopping <- callback_early_stopping(monitor = "val_loss", 
                                          patience = 10,    # 验证集损失在连续10轮内没有改进则停止
                                          restore_best_weights = TRUE)


model_lstm %>% fit(
  X_processed, 
  y_train, 
  epochs = 30,
  batch_size = 32, 
  validation_split = 0.1,
  callbacks = list(lr_scheduler,early_stopping)
)


y_pred_lstm <- predict(model_lstm,X_test_processed)
rmse <- sqrt(mean((y_test - y_pred_lstm)^2))
cat("RMSE: ", rmse, "\n")
rss <- sum((y_test - y_pred_lstm)^2)  # Residual sum of squares
tss <- sum((y_test - mean(y_test))^2)  # Total sum of squares
r2 <- 1 - (rss / tss)
cat("R²: ", r2, "\n")