

library(dplyr)
library(tidyr)
library(keras3)

# 1. 基础加载
tianjinall <- read.csv2("tianjingall.txt", sep = "\t", header = TRUE)
tianjinall$DATE <- as.Date(tianjinall$DATE)
tianjinall[, 2:6] <- lapply(tianjinall[, 2:6], as.numeric)

# 2. 核心预处理：按 mcode 分组处理滞后 (防止数据跨点污染)
trantianjing <- tianjinall %>%
  # 必须先按站点和日期排序
  arrange(mcode, DATE) %>%
  group_by(mcode) %>%
  mutate(
    across(c(huxi, tmean, rhmean, WHA2_res), ~lag(., 1), .names = "{col}_lag1"),
    across(c(huxi, tmean, rhmean, WHA2_res), ~lag(., 2), .names = "{col}_lag2"),
    across(c(huxi, tmean, rhmean, WHA2_res), ~lag(., 3), .names = "{col}_lag3")
  ) %>%
  ungroup() %>%
  drop_na() # 剔除每个 mcode 前 3 行产生的 NA

# 3. 提取特征与标签
# 选取的 6 列：huxi_lag1, WHA2_res_lag1, huxi_lag2, WHA2_res_lag2, huxi_lag3, WHA2_res_lag3
X_mat <- as.matrix(trantianjing[, c("huxi_lag1", "WHA2_res_lag1", 
                                    "huxi_lag2", "WHA2_res_lag2", 
                                    "huxi_lag3", "WHA2_res_lag3")])
y_vec <- as.numeric(trantianjing$huxi)

# 4. 考虑到多站点，采样建议 (可以使用随机拆分，因为 lag 已在分组内完成)
# 1. 确定切分时间点（例如将最后 20% 的日期作为测试集）
all_dates <- sort(unique(trantianjing$DATE))
split_point <- all_dates[floor(length(all_dates) * 0.8)]

# 2. 按日期物理隔绝训练集与测试集
train_data <- trantianjing %>% filter(DATE <= split_point)
test_data  <- trantianjing %>% filter(DATE > split_point)

# 3. 提取矩阵
X_train_raw <- as.matrix(train_data[, c("huxi_lag1", "WHA2_res_lag1", "huxi_lag2", "WHA2_res_lag2", "huxi_lag3", "WHA2_res_lag3")])
y_train_raw <- as.numeric(train_data$huxi)

X_test_raw  <- as.matrix(test_data[, c("huxi_lag1", "WHA2_res_lag1", "huxi_lag2", "WHA2_res_lag2", "huxi_lag3", "WHA2_res_lag3")])
y_test_raw  <- as.numeric(test_data$huxi)

# 4. 严格标准化：仅使用训练集的均值和标准差
train_mean_X <- colMeans(X_train_raw)
train_sd_X   <- apply(X_train_raw, 2, sd)

X_train_scaled <- scale(X_train_raw, center = train_mean_X, scale = train_sd_X)
# 注意：测试集必须用训练集的参数来 scale，绝对不能用自己的参数
X_test_scaled  <- scale(X_test_raw,  center = train_mean_X, scale = train_sd_X)

# y 同理
mean_y <- mean(y_train_raw)
sd_y   <- sd(y_train_raw)
y_train_scaled <- (y_train_raw - mean_y) / sd_y
# 5. 标准化与 3D 转换 (同前)
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

# 6. 训练模型 (建议增加 Epoch)
model_lstm <- keras_model_sequential() %>%
  layer_lstm(units = 64, input_shape = c(3, 2)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)

model_lstm %>% compile(loss = 'mse', optimizer = 'adam')

model_lstm %>% fit(X_train_arr, y_train_scaled, epochs = 50, batch_size = 32, validation_split = 0.1, verbose = 1)

# 7. 计算 R2
y_pred_scaled <- predict(model_lstm, X_test_arr)
y_pred_final  <- (as.numeric(y_pred_scaled) * sd_y) + mean_y

r2 <- 1 - (sum((y_test_raw - y_pred_final)^2) / sum((y_test_raw - mean(y_test_raw))^2))
cat("最终 R²: ", r2, "\n")


# 将预测结果合并回测试集数据框
plot_df <- test_data %>%
  mutate(Predicted = y_pred_final) %>%
  select(DATE, mcode, Actual = huxi, Predicted)

# 方案 A：选取发病量最大的一个站点进行展示（最清晰）
top_mcode <- plot_df %>% 
  group_by(mcode) %>% 
  summarise(total = sum(Actual)) %>% 
  arrange(desc(total)) %>% 
  slice(1) %>% 
  pull(mcode)

single_site_df <- plot_df %>% filter(mcode == top_mcode)

# 方案 B：计算所有站点的日平均值（展示整体趋势）
trend_df <- plot_df %>%
  group_by(DATE) %>%
  summarise(Actual = mean(Actual), Predicted = mean(Predicted))

library(ggplot2)
library(scales) # 用于处理日期格式

p_trend <- ggplot(trend_df, aes(x = DATE)) +
  # 真实值：灰色区域或细线
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 1, alpha = 0.7) +
  # 预测值：蓝色粗线
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 1) +
  # 装饰
  scale_color_manual(values = c("Actual" = "#999999", "Predicted" = "#0072B2")) +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "2 months") +
  theme_minimal(base_size = 16) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  labs(
    #title = paste("站点平均门诊量预测对比 (测试集)"),
    subtitle = paste(" R² =", round(r2, 3)," RMSE =", round(rmse, 3)),
    x = "Data",
    y = "Daily outpatient volume (number of visits)",
    color = ""
  )

print(p_trend)
