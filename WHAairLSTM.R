

library(dplyr)
library(tidyr)
library(keras3)

####Calculate wha
cq <- cq %>%
  mutate(
    pm25_exposure =(exp(0.00013578 * pm25)-1)*100,
    pm10_exposure =(exp(0.000222504 * pm10)-1)*100,
    no2_exposure = (exp(0.001391996 * no2)-1)*100,
    so2_exposure = (exp(0.00220782 * so2)-1)*100,
    o3_exposure = (exp(0.000770138 * o3)-1)*100
  ) %>%

  mutate(er_rescir_all = rowSums(select(., ends_with("_exposure"))))



cq <- cq[, c(1:3, 10:19, 48)]
cq$date <- as.Date(cq$date)
cqdata2 <- cq %>%
  filter(format(date, "%Y") >= "2018")

wha1 <- c(9.,13.36,20.28,25.41)
wha2 <- c(9.15,13.39,17.98,29.73)
wha3 <- c(8.32,16.63,21.99,33.27)
wha2 <- c(8.94,12.99,17.95,31.82)


#####wha2
cqdata2$wha2<-NA
cqdata2$wha2[cqdata2$er_rescir_all<wha2[1]]<-1
cqdata2$wha2[cqdata2$er_rescir_all>wha2[1]      & cqdata2$er_rescir_all<=wha2[2]]<-2
cqdata2$wha2[cqdata2$er_rescir_all>wha2[2]   & cqdata2$er_rescir_all<=wha2[3]]<-3
cqdata2$wha2[cqdata2$er_rescir_all>wha2[3]     & cqdata2$er_rescir_all<=wha2[4]]<-4
cqdata2$wha2[cqdata2$er_rescir_all>wha2[4]]<-5
summary(factor(cqdata2$wha2))

######pop#####
read.csv2("../pop.txt",sep = "\t",header = T)->pop
colnames(pop) <- c("mcode","county","pop")
cqdata2 <- merge(cqdata2,pop,by = "county")


cqdata2$DOW <- as.POSIXlt(cqdata2$date)$wday
cqdata2$year <- as.POSIXlt(cqdata2$date)$year + 1900
cqdata2$month <- as.POSIXlt(cqdata2$date)$mon + 1
cqdata2$Stratum <- as.factor(factor(cqdata2$mcode):factor(cqdata2$year):factor(cqdata2$month):factor(cqdata2$DOW))

########gnm model ######
coln <- c("非意外疾病总量","呼吸系统疾病","循环系统疾病","缺血性心脏病","中风","下呼吸道感染","慢性阻塞性肺病","哮喘") 

plotss <- list()
for (i in 1:length(coln)){
      model <- gnm(cqdata2[,(i+5)] ~ factor(cqdata2$wha2)+ 
                 ns(cqdata2$tem_avg, df=3) + ns(cqdata2$rhu_avg, df=3),
               family = quasipoisson,
               eliminate = factor(cqdata2$Stratum),
               offset=log(cqdata2$pop), 
               data=cqdata2)
  
  summary(model)
  output <- as.data.frame(summary(model)$coefficients)
  output$CoefName <- rownames(summary(model)$coefficients)
  output<- output[1:4,]
  output$CoefName <- sub(".*\\(([^)]+)\\)(\\d+)", "\\2", output$CoefName)
  #results[[col_name]] <- output


tianjinall <- read.csv2("tianjingall.txt", sep = "\t", header = TRUE)
tianjinall$DATE <- as.Date(tianjinall$DATE)
tianjinall[, 2:6] <- lapply(tianjinall[, 2:6], as.numeric)


trantianjing <- tianjinall %>%
  arrange(mcode, DATE) %>%
  group_by(mcode) %>%
  mutate(
    across(c(huxi, tmean, rhmean, WHA2_res), ~lag(., 1), .names = "{col}_lag1"),
    across(c(huxi, tmean, rhmean, WHA2_res), ~lag(., 2), .names = "{col}_lag2"),
    across(c(huxi, tmean, rhmean, WHA2_res), ~lag(., 3), .names = "{col}_lag3")
  ) %>%
  ungroup() %>%
  drop_na() 


X_mat <- as.matrix(trantianjing[, c("huxi_lag1", "WHA2_res_lag1", 
                                    "huxi_lag2", "WHA2_res_lag2", 
                                    "huxi_lag3", "WHA2_res_lag3")])
y_vec <- as.numeric(trantianjing$huxi)


all_dates <- sort(unique(trantianjing$DATE))
split_point <- all_dates[floor(length(all_dates) * 0.8)]


train_data <- trantianjing %>% filter(DATE <= split_point)
test_data  <- trantianjing %>% filter(DATE > split_point)


X_train_raw <- as.matrix(train_data[, c("huxi_lag1", "WHA2_res_lag1", "huxi_lag2", "WHA2_res_lag2", "huxi_lag3", "WHA2_res_lag3")])
y_train_raw <- as.numeric(train_data$huxi)

X_test_raw  <- as.matrix(test_data[, c("huxi_lag1", "WHA2_res_lag1", "huxi_lag2", "WHA2_res_lag2", "huxi_lag3", "WHA2_res_lag3")])
y_test_raw  <- as.numeric(test_data$huxi)


train_mean_X <- colMeans(X_train_raw)
train_sd_X   <- apply(X_train_raw, 2, sd)

X_train_scaled <- scale(X_train_raw, center = train_mean_X, scale = train_sd_X)

X_test_scaled  <- scale(X_test_raw,  center = train_mean_X, scale = train_sd_X)


mean_y <- mean(y_train_raw)
sd_y   <- sd(y_train_raw)
y_train_scaled <- (y_train_raw - mean_y) / sd_y

mean_y <- mean(y_train_raw); sd_y <- sd(y_train_raw)
y_train_scaled <- (y_train_raw - mean_y) / sd_y

train_mean_X <- colMeans(X_train_raw); train_sd_X <- apply(X_train_raw, 2, sd)
X_train_scaled <- scale(X_train_raw, center = train_mean_X, scale = train_sd_X)
X_test_scaled  <- scale(X_test_raw,  center = train_mean_X, scale = train_sd_X)

build_lstm_data <- function(X_scaled_matrix) {
  n <- nrow(X_scaled_matrix)
  arr <- array(0, dim = c(n, 3, 2))
  for (i in 1:n) {
    arr[i, 1, ] <- X_scaled_matrix[i, 5:6] # Lag 3
    arr[i, 2, ] <- X_scaled_matrix[i, 3:4] # Lag 2
    arr[i, 3, ] <- X_scaled_matrix[i, 1:2] # Lag 1
  }
  return(arr)
}

X_train_arr <- build_lstm_data(as.matrix(X_train_scaled))
X_test_arr  <- build_lstm_data(as.matrix(X_test_scaled))

model_lstm <- keras_model_sequential() %>%
  layer_lstm(units = 64, input_shape = c(3, 2)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)

model_lstm %>% compile(loss = 'mse', optimizer = 'adam')

model_lstm %>% fit(X_train_arr, y_train_scaled, epochs = 50, batch_size = 32, validation_split = 0.1, verbose = 1)


y_pred_scaled <- predict(model_lstm, X_test_arr)
y_pred_final  <- (as.numeric(y_pred_scaled) * sd_y) + mean_y

r2 <- 1 - (sum((y_test_raw - y_pred_final)^2) / sum((y_test_raw - mean(y_test_raw))^2))




plot_df <- test_data %>%
  mutate(Predicted = y_pred_final) %>%
  select(DATE, mcode, Actual = huxi, Predicted)


top_mcode <- plot_df %>% 
  group_by(mcode) %>% 
  summarise(total = sum(Actual)) %>% 
  arrange(desc(total)) %>% 
  slice(1) %>% 
  pull(mcode)

single_site_df <- plot_df %>% filter(mcode == top_mcode)


trend_df <- plot_df %>%
  group_by(DATE) %>%
  summarise(Actual = mean(Actual), Predicted = mean(Predicted))

library(ggplot2)
library(scales)

p_trend <- ggplot(trend_df, aes(x = DATE)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 1, alpha = 0.7) +
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 1) +
  scale_color_manual(values = c("Actual" = "#999999", "Predicted" = "#0072B2")) +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "2 months") +
  theme_minimal(base_size = 16) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  labs(
    subtitle = paste(" R² =", round(r2, 3)," RMSE =", round(rmse, 3)),
    x = "Data",
    y = "Daily outpatient volume (number of visits)",
    color = ""
  )
