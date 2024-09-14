# Load necessary libraries----
library(VIM)
library(scales)
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ggplot2)
library(class)
library(glmnet)
library(caret)

# MEAN IMPUTATION----
cat("Pre processing: Mean imputation.\n")
dataset_mean_impute = read.csv('Life Expectency data.csv')

dataset_mean_impute$Adult.Mortality = ifelse(is.na(dataset_mean_impute$Adult.Mortality),ave(dataset_mean_impute$Adult.Mortality, FUN = function(x) mean(x, na.rm = TRUE)),dataset_mean_impute$Adult.Mortality)
dataset_mean_impute$Alcohol = ifelse(is.na(dataset_mean_impute$Alcohol),ave(dataset_mean_impute$Alcohol, FUN = function(x) mean(x, na.rm = TRUE)),dataset_mean_impute$Alcohol)
dataset_mean_impute$Life.expectancy = ifelse(is.na(dataset_mean_impute$Life.expectancy),ave(dataset_mean_impute$Life.expectancy, FUN = function(x) mean(x, na.rm = TRUE)),dataset_mean_impute$Life.expectancy)
dataset_mean_impute$Hepatitis.B = ifelse(is.na(dataset_mean_impute$Hepatitis.B),ave(dataset_mean_impute$Hepatitis.B, FUN = function(x) mean(x, na.rm = TRUE)),dataset_mean_impute$Hepatitis.B)
dataset_mean_impute$Total.expenditure= ifelse(is.na(dataset_mean_impute$Total.expenditure),ave(dataset_mean_impute$Total.expenditure, FUN = function(x) mean(x, na.rm = TRUE)),dataset_mean_impute$Total.expenditure)
dataset_mean_impute$GDP= ifelse(is.na(dataset_mean_impute$GDP),ave(dataset_mean_impute$GDP, FUN = function(x) mean(x, na.rm = TRUE)),dataset_mean_impute$GDP)
dataset_mean_impute$Population= ifelse(is.na(dataset_mean_impute$Population),ave(dataset_mean_impute$Population, FUN = function(x) mean(x, na.rm = TRUE)),dataset_mean_impute$Population)
dataset_mean_impute$thinness..1.19.years = ifelse(is.na(dataset_mean_impute$thinness..1.19.years), ave(dataset_mean_impute$thinness..1.19.years, FUN = function(x) mean(x, na.rm = TRUE)),dataset_mean_impute$thinness..1.19.years)
dataset_mean_impute$thinness.5.9.years = ifelse(is.na(dataset_mean_impute$thinness.5.9.years), ave(dataset_mean_impute$thinness.5.9.years, FUN = function(x) mean(x, na.rm = TRUE)),dataset_mean_impute$thinness.5.9.years)
dataset_mean_impute$Income.composition.of.resources = ifelse(is.na(dataset_mean_impute$Income.composition.of.resources), ave(dataset_mean_impute$Income.composition.of.resources, FUN = function(x) mean(x, na.rm = TRUE)),dataset_mean_impute$Income.composition.of.resources)
dataset_mean_impute$Schooling= ifelse(is.na(dataset_mean_impute$Schooling), ave(dataset_mean_impute$Schooling, FUN = function(x) mean(x, na.rm = TRUE)), dataset_mean_impute$Schooling)
dataset_mean_impute$BMI= ifelse(is.na(dataset_mean_impute$BMI), ave(dataset_mean_impute$BMI, FUN = function(x) mean(x, na.rm = TRUE)), dataset_mean_impute$BMI)
dataset_mean_impute$Diphtheria= ifelse(is.na(dataset_mean_impute$Diphtheria), ave(dataset_mean_impute$Diphtheria, FUN = function(x) mean(x, na.rm = TRUE)), dataset_mean_impute$Diphtheria)
dataset_mean_impute$Polio= ifelse(is.na(dataset_mean_impute$Polio), ave(dataset_mean_impute$Polio, FUN = function(x) mean(x, na.rm = TRUE)), dataset_mean_impute$Polio)
#Normalization
dataset_mean_impute[,4:22] <- apply(dataset_mean_impute[,4:22], 2, rescale)
# Data pre-processing----
set.seed(1234)
categorical_column=dataset_mean_impute$Country
categorical_column_encoded=as.factor(categorical_column)
dataset_mean_impute$Country=categorical_column_encoded
categorical_column1=dataset_mean_impute$Status
categorical_column_encoded1=as.factor(categorical_column1)
dataset_mean_impute$Status=categorical_column_encoded1
categorical_column2=dataset_mean_impute$Year
categorical_column_encoded2=as.factor(categorical_column2)
dataset_mean_impute$Year=categorical_column_encoded2

set.seed(123)
split <- sample.split(dataset_mean_impute$Life.expectancy, SplitRatio = 0.8)
train_data <- dataset_mean_impute[split, ]
test_data <- dataset_mean_impute[!split, ]

# Exclude the first and fourth columns from the formula
formula <- Life.expectancy ~ . - Country
tree_model <- rpart(formula, data = train_data, control = rpart.control(minsplit = 4))

# Decision tree----
# Fit the unpruned decision tree model
tree_model_unpruned <- rpart(Life.expectancy ~ ., data = train_data, control = rpart.control(minsplit = 4, cp = 0.01))

# Make predictions with the unpruned tree
predictions_tree_unpruned <- predict(tree_model_unpruned, newdata = test_data)

# Calculating R-squared, RMSE, and MSE for the unpruned Decision Tree
actual_values_tree_unpruned <- test_data$Life.expectancy
r_squared_tree_unpruned <- 1 - sum((actual_values_tree_unpruned - as.vector(predictions_tree_unpruned))^2) / sum((actual_values_tree_unpruned - mean(actual_values_tree_unpruned))^2)
rmse_tree_unpruned <- sqrt(mean((actual_values_tree_unpruned - as.vector(predictions_tree_unpruned))^2))
mse_tree_unpruned <- mean((actual_values_tree_unpruned - as.vector(predictions_tree_unpruned))^2)

# Print the metrics for the unpruned Decision Tree
cat("Decision Tree Metrics (Unpruned):\n")
cat("R-squared:", round(r_squared_tree_unpruned, 4), "\n")
cat("RMSE:", rmse_tree_unpruned, "\n")
cat("MSE:", mse_tree_unpruned, "\n")

# Prune the tree
pruned_tree_model <- prune(tree_model, cp = 0.02)

# Make predictions with the pruned tree
predictions_pruned_tree <- predict(pruned_tree_model, newdata = test_data)

# Calculating R-squared, RMSE, and MSE for the pruned Decision Tree
actual_values_tree <- test_data$Life.expectancy
predictions_tree <- predict(pruned_tree_model, newdata = test_data)
r_squared_tree <- 1 - sum((actual_values_tree - predictions_tree)^2) / sum((actual_values_tree - mean(actual_values_tree))^2)
rmse_tree <- sqrt(mean((actual_values_tree - predictions_tree)^2))
mse_tree <- mean((actual_values_tree - predictions_tree)^2)

# Print the metrics for the pruned Decision Tree
cat("\nDecision Tree Metrics (Pruned):\n")
cat("R-squared:", round(r_squared_tree, 4), "\n")
cat("RMSE:", rmse_tree, "\n")
cat("MSE:", mse_tree, "\n")

# Random forest----
target_rf <- dataset_mean_impute$Life.expectancy
predictors_rf <- dataset_mean_impute[, -c(1, 4)]

# Data splitting
split_rf <- sample.split(target_rf, SplitRatio = 0.8)
train_data_rf <- predictors_rf[split_rf, ]
train_target_rf <- target_rf[split_rf]
test_data_rf <- predictors_rf[!split_rf, ]
test_target_rf <- target_rf[!split_rf]

# Train the Random Forest model
rf_model <- randomForest(x = train_data_rf, y = train_target_rf, ntree = 500)

# Predictions for Random Forest
predictions_rf <- predict(rf_model, test_data_rf)
# ... (visualization code remains the same)

# Calculating R-squared, RMSE, and MSE for Random Forest
actual_values_rf <- test_target_rf
r_squared_rf <- 1 - sum((actual_values_rf - predictions_rf)^2) / sum((actual_values_rf - mean(actual_values_rf))^2)
rmse_rf <- sqrt(mean((actual_values_rf - predictions_rf)^2))
mse_rf <- mean((actual_values_rf - predictions_rf)^2)

# Print the metrics for the Random Forest
cat("\n\nRandom Forest Metrics:\n")
cat("R-squared:", round(r_squared_rf, 4), "\n")
cat("RMSE:", rmse_rf, "\n")
cat("MSE:", mse_rf, "\n")

# Elastic Net Regression----
# Data pre-processing
set.seed(123)
# Specify the column names correctly
target_column <- "Life.expectancy"
# Exclude the first four columns
predictor_columns <- names(train_data)[5:ncol(train_data)]

# Convert data frames to matrices
X_train <- as.matrix(train_data[, predictor_columns])
y_train <- as.numeric(train_data[, target_column])
X_test <- as.matrix(test_data[, predictor_columns])
y_test <- as.numeric(test_data[, target_column])  # Include y_test

# Fit an Elastic Net regression model
alpha <- 0.5  # Adjust the alpha parameter (0 for Ridge, 1 for Lasso)
enet_model <- glmnet(X_train, y_train, alpha = alpha)

# Perform cross-validation to find the optimal lambda
cv_model <- cv.glmnet(X_train, y_train, alpha = alpha)

# Get the optimal lambda
optimal_lambda <- cv_model$lambda.min

# Get the coefficients for the optimal lambda
coefficients(enet_model, s = optimal_lambda)

# Make predictions on the test data
predictions <- predict(enet_model, s = optimal_lambda, newx = X_test)

# Calculate R-squared for Elastic Net regression
mean_actual_values <- mean(y_test)
sse <- sum((y_test - as.vector(predictions))^2)
sst <- sum((y_test - mean_actual_values)^2)
r_squared_enet <- 1 - (sse / sst)

# Calculate RMSE for Elastic Net regression
rmse_enet <- sqrt(mean((y_test - as.vector(predictions))^2))

# Calculate MSE for Elastic Net regression
mse_enet <- mean((y_test - as.vector(predictions))^2)

# Print the metrics for Elastic Net Regression
cat("\n\nElastic Net Regression Metrics:\n")
cat("R-squared:", round(r_squared_enet, 4), "\n")
cat("RMSE:", rmse_enet, "\n")
cat("MSE:",mse_enet,"\n")

# Define the target variable
target_column <- "Life.expectancy"

# Remove the first four columns (columns 1 to 4) from the dataset_mean_impute
dataset_mean_impute <- dataset_mean_impute[, -c(1:3)]

# Split the data into training and testing sets (80% train, 20% test)
set.seed(123)
split <- sample.split(dataset_mean_impute[, target_column], SplitRatio = 0.8)
train_data <- dataset_mean_impute[split, ]
test_data <- dataset_mean_impute[!split, ]

# Fit an Elastic Net regression model
alpha <- 0.5  # Adjust the alpha parameter (0 for Ridge, 1 for Lasso)
enet_model <- glmnet(as.matrix(train_data[, -which(names(train_data) == target_column)]), train_data[, target_column], alpha = alpha)

# Perform cross-validation to find the optimal lambda
cv_model <- cv.glmnet(as.matrix(train_data[, -which(names(train_data) == target_column)]), train_data[, target_column], alpha = alpha)

# Get the optimal lambda
optimal_lambda <- cv_model$lambda.min

# Get the coefficients for the optimal lambda
coefficients(enet_model, s = optimal_lambda)

# Make predictions on the test data
predictions <- predict(enet_model, s = optimal_lambda, newx = as.matrix(test_data[, -which(names(test_data) == target_column)]))

# Calculate R-squared for Elastic Net regression
mean_actual_values <- mean(test_data[, target_column])
sse <- sum((test_data[, target_column] - as.vector(predictions))^2)
sst <- sum((test_data[, target_column] - mean_actual_values)^2)
r_squared_enet <- 1 - (sse / sst)

# Calculate RMSE for Elastic Net regression
rmse_enet <- sqrt(mean((test_data[, target_column] - as.vector(predictions))^2))

# Calculate MSE for Elastic Net regression
mse_enet <- mean((test_data[, target_column] - as.vector(predictions))^2)

# Print the metrics for Elastic Net Regression
cat("\nElastic Net Regression Metrics after Feature Selection:\n")
cat("R-squared:", round(r_squared_enet, 4), "\n")
cat("RMSE:", rmse_enet, "\n")
cat("MSE:", mse_enet, "\n")

# MLR----
# Remove the first 3 rows from the dataset
dataset_mean_impute <- dataset_mean_impute[-c(1:3), ]
# Fit a Multivariate Linear Regression (MLR) model without feature selection
mlr_model_without_fs <- lm(formula = paste(target_column, "~ ."), data = train_data)

# Make predictions on the test data
predictions_without_fs <- predict(mlr_model_without_fs, newdata = test_data)

# Calculate R-squared for MLR without feature selection
mean_actual_values <- mean(test_data[, target_column])
sse_without_fs <- sum((test_data[, target_column] - predictions_without_fs)^2)
sst_without_fs <- sum((test_data[, target_column] - mean_actual_values)^2)
r_squared_mlr_without_fs <- 1 - (sse_without_fs / sst_without_fs)

# Calculate RMSE for MLR without feature selection
rmse_mlr_without_fs <- sqrt(mean((test_data[, target_column] - predictions_without_fs)^2))

# Calculate MSE for MLR without feature selection
mse_mlr_without_fs <- mean((test_data[, target_column] - predictions_without_fs)^2)

# Print the metrics for Multivariate Linear Regression (MLR) without feature selection
cat("\n\nMultivariate Linear Regression (MLR) Metrics without Feature Selection:\n")
cat("R-squared:", round(r_squared_mlr_without_fs, 4), "\n")
cat("RMSE:", rmse_mlr_without_fs, "\n")
cat("MSE:", mse_mlr_without_fs, "\n")

# Multivariate Linear Regression (MLR) with Lasso Feature Selection

# Fit a Multivariate Linear Regression (MLR) model with Lasso feature selection
mlr_lasso_model <- glmnet(as.matrix(train_data[, predictor_columns]), train_data[, target_column], alpha = 1)

# Perform cross-validation to find the optimal lambda (alpha = 1 for Lasso)
cv_lasso_model <- cv.glmnet(as.matrix(train_data[, predictor_columns]), train_data[, target_column], alpha = 1)

# Get the optimal lambda for Lasso
optimal_lambda_lasso <- cv_lasso_model$lambda.min

# Fit the final Lasso model with the optimal lambda
final_lasso_model <- glmnet(as.matrix(train_data[, predictor_columns]), train_data[, target_column], alpha = 1, lambda = optimal_lambda_lasso)

# Make predictions on the test data with the Lasso-selected features
predictions_lasso <- predict(final_lasso_model, s = optimal_lambda_lasso, newx = as.matrix(test_data[, predictor_columns]))

# Calculate R-squared for Lasso-selected features
mean_actual_values <- mean(test_data[, target_column])
sse_lasso <- sum((test_data[, target_column] - as.vector(predictions_lasso))^2)
sst_lasso <- sum((test_data[, target_column] - mean_actual_values)^2)
r_squared_lasso <- 1 - (sse_lasso / sst_lasso)

# Calculate RMSE for Lasso-selected features
rmse_lasso <- sqrt(mean((test_data[, target_column] - as.vector(predictions_lasso))^2))

# Calculate MSE for Lasso-selected features
mse_lasso <- mean((test_data[, target_column] - as.vector(predictions_lasso))^2)

# Print the metrics for Multivariate Linear Regression (MLR) with Lasso feature selection
cat("\nMultivariate Linear Regression (MLR) Metrics with Lasso Feature Selection:\n")
cat("R-squared:", round(r_squared_lasso, 4), "\n")
cat("RMSE:", rmse_lasso, "\n")
cat("MSE:", mse_lasso, "\n")



#HOTDECK----
cat("\n\n\nPre processing: HotDeck.")
dataset_hot_deck = read.csv('Life Expectency data.csv')

# Perform hot deck imputation for each column with missing values----
columns_with_missing_values <- c(
  "Adult.Mortality", "Alcohol", "Life.expectancy", "Hepatitis.B",
  "Total.expenditure", "GDP", "Population", "thinness..1.19.years",
  "thinness.5.9.years", "Income.composition.of.resources",
  "Schooling", "BMI", "Diphtheria", "Polio"
)

for (column in columns_with_missing_values) {
  dataset_hot_deck <- hotdeck(dataset_hot_deck, variable = column)
}

# Normalization----
dataset_hot_deck[,4:22] <- apply(dataset_hot_deck[,4:22], 2, rescale)

# Data pre-processing----
set.seed(123)
categorical_column=dataset_hot_deck$Country
categorical_column_encoded=as.factor(categorical_column)
dataset_hot_deck$Country=categorical_column_encoded
categorical_column1=dataset_hot_deck$Status
categorical_column_encoded1=as.factor(categorical_column1)
dataset_hot_deck$Status=categorical_column_encoded1
categorical_column2=dataset_hot_deck$Year
categorical_column_encoded2=as.factor(categorical_column2)
dataset_hot_deck$Year=categorical_column_encoded2

set.seed(123)
split <- sample.split(dataset_hot_deck$Life.expectancy, SplitRatio = 0.8)
train_data <- dataset_hot_deck[split, ]
test_data <- dataset_hot_deck[!split, ]


# Exclude the first and fourth columns from the formula
formula <- Life.expectancy ~ . - Country
tree_model <- rpart(formula, data = train_data, control = rpart.control(minsplit = 4))

# Decision tree----
# Fit the unpruned decision tree model
tree_model_unpruned <- rpart(Life.expectancy ~ ., data = train_data, control = rpart.control(minsplit = 4, cp = 0.01))

# Make predictions with the unpruned tree
predictions_tree_unpruned <- predict(tree_model_unpruned, newdata = test_data)

# Calculating R-squared, RMSE, and MSE for the unpruned Decision Tree
actual_values_tree_unpruned <- test_data$Life.expectancy
r_squared_tree_unpruned <- 1 - sum((actual_values_tree_unpruned - as.vector(predictions_tree_unpruned))^2) / sum((actual_values_tree_unpruned - mean(actual_values_tree_unpruned))^2)
rmse_tree_unpruned <- sqrt(mean((actual_values_tree_unpruned - as.vector(predictions_tree_unpruned))^2))
mse_tree_unpruned <- mean((actual_values_tree_unpruned - as.vector(predictions_tree_unpruned))^2)

# Print the metrics for the unpruned Decision Tree
cat("\nDecision Tree Metrics (Unpruned):\n")
cat("R-squared:", round(r_squared_tree_unpruned, 4), "\n")
cat("RMSE:", rmse_tree_unpruned, "\n")
cat("MSE:", mse_tree_unpruned, "\n")

# Prune the tree
pruned_tree_model <- prune(tree_model, cp = 0.02)

# Make predictions with the pruned tree
predictions_pruned_tree <- predict(pruned_tree_model, newdata = test_data)

# Calculating R-squared, RMSE, and MSE for the pruned Decision Tree
actual_values_tree <- test_data$Life.expectancy
predictions_tree <- predict(pruned_tree_model, newdata = test_data)
r_squared_tree <- 1 - sum((actual_values_tree - predictions_tree)^2) / sum((actual_values_tree - mean(actual_values_tree))^2)
rmse_tree <- sqrt(mean((actual_values_tree - predictions_tree)^2))
mse_tree <- mean((actual_values_tree - predictions_tree)^2)

# Print the metrics for the pruned Decision Tree
cat("\nDecision Tree Metrics (Pruned):\n")
cat("R-squared:", round(r_squared_tree, 4), "\n")
cat("RMSE:", rmse_tree, "\n")
cat("MSE:", mse_tree, "\n")


# Random forest----
target_rf <- dataset_hot_deck$Life.expectancy
predictors_rf <- dataset_hot_deck[, -c(1, 4)]

# Data splitting
split_rf <- sample.split(target_rf, SplitRatio = 0.8)
train_data_rf <- predictors_rf[split_rf, ]
train_target_rf <- target_rf[split_rf]
test_data_rf <- predictors_rf[!split_rf, ]
test_target_rf <- target_rf[!split_rf]

# Train the Random Forest model
rf_model <- randomForest(x = train_data_rf, y = train_target_rf, ntree = 500)

# Predictions for Random Forest
predictions_rf <- predict(rf_model, test_data_rf)
# ... (visualization code remains the same)

# Calculating R-squared, RMSE, and MSE for Random Forest
actual_values_rf <- test_target_rf
r_squared_rf <- 1 - sum((actual_values_rf - predictions_rf)^2) / sum((actual_values_rf - mean(actual_values_rf))^2)
rmse_rf <- sqrt(mean((actual_values_rf - predictions_rf)^2))
mse_rf <- mean((actual_values_rf - predictions_rf)^2)

# Print the metrics for the Random Forest
cat("\n\nRandom Forest Metrics:\n")
cat("R-squared:", round(r_squared_rf, 4), "\n")
cat("RMSE:", rmse_rf, "\n")
cat("MSE:", mse_rf, "\n")

# Elastic Net Regression----
# Data pre-processing
set.seed(123)
# Specify the column names correctly
target_column <- "Life.expectancy"
# Exclude the first four columns
predictor_columns <- names(train_data)[5:ncol(train_data)]

# Convert data frames to matrices
X_train <- as.matrix(train_data[, predictor_columns])
y_train <- as.numeric(train_data[, target_column])
X_test <- as.matrix(test_data[, predictor_columns])
y_test <- as.numeric(test_data[, target_column])  # Include y_test

# Fit an Elastic Net regression model
alpha <- 0.5  # Adjust the alpha parameter (0 for Ridge, 1 for Lasso)
enet_model <- glmnet(X_train, y_train, alpha = alpha)

# Perform cross-validation to find the optimal lambda
cv_model <- cv.glmnet(X_train, y_train, alpha = alpha)

# Get the optimal lambda
optimal_lambda <- cv_model$lambda.min

# Get the coefficients for the optimal lambda
coefficients(enet_model, s = optimal_lambda)

# Make predictions on the test data
predictions <- predict(enet_model, s = optimal_lambda, newx = X_test)

# Calculate R-squared for Elastic Net regression
mean_actual_values <- mean(y_test)
sse <- sum((y_test - as.vector(predictions))^2)
sst <- sum((y_test - mean_actual_values)^2)
r_squared_enet <- 1 - (sse / sst)

# Calculate RMSE for Elastic Net regression
rmse_enet <- sqrt(mean((y_test - as.vector(predictions))^2))

# Calculate MSE for Elastic Net regression
mse_enet <- mean((y_test - as.vector(predictions))^2)

# Print the metrics for Elastic Net Regression
cat("\n\nElastic Net Regression Metrics:\n")
cat("R-squared:", round(r_squared_enet, 4), "\n")
cat("RMSE:", rmse_enet, "\n")
cat("MSE:",mse_enet,"\n")

# Define the target variable
target_column <- "Life.expectancy"

# Remove the first four columns (columns 1 to 4) from the dataset_hot_deck
dataset_hot_deck <- dataset_hot_deck[, -c(1:3)]

# Split the data into training and testing sets (80% train, 20% test)
set.seed(123)
split <- sample.split(dataset_hot_deck[, target_column], SplitRatio = 0.8)
train_data <- dataset_hot_deck[split, ]
test_data <- dataset_hot_deck[!split, ]

# Fit an Elastic Net regression model
alpha <- 0.5  # Adjust the alpha parameter (0 for Ridge, 1 for Lasso)
enet_model <- glmnet(as.matrix(train_data[, -which(names(train_data) == target_column)]), train_data[, target_column], alpha = alpha)

# Perform cross-validation to find the optimal lambda
cv_model <- cv.glmnet(as.matrix(train_data[, -which(names(train_data) == target_column)]), train_data[, target_column], alpha = alpha)

# Get the optimal lambda
optimal_lambda <- cv_model$lambda.min

# Get the coefficients for the optimal lambda
coefficients(enet_model, s = optimal_lambda)

# Make predictions on the test data
predictions <- predict(enet_model, s = optimal_lambda, newx = as.matrix(test_data[, -which(names(test_data) == target_column)]))

# Calculate R-squared for Elastic Net regression
mean_actual_values <- mean(test_data[, target_column])
sse <- sum((test_data[, target_column] - as.vector(predictions))^2)
sst <- sum((test_data[, target_column] - mean_actual_values)^2)
r_squared_enet <- 1 - (sse / sst)

# Calculate RMSE for Elastic Net regression
rmse_enet <- sqrt(mean((test_data[, target_column] - as.vector(predictions))^2))

# Calculate MSE for Elastic Net regression
mse_enet <- mean((test_data[, target_column] - as.vector(predictions))^2)

# Print the metrics for Elastic Net Regression
cat("\nElastic Net Regression Metrics after Feature Selection:\n")
cat("R-squared:", round(r_squared_enet, 4), "\n")
cat("RMSE:", rmse_enet, "\n")
cat("MSE:", mse_enet, "\n")

# MLR----
# Remove the first 3 rows from the dataset
dataset_mean_impute <- dataset_hot_deck[-c(1:3), ]
# Fit a Multivariate Linear Regression (MLR) model without feature selection
mlr_model_without_fs <- lm(formula = paste(target_column, "~ ."), data = train_data)

# Make predictions on the test data
predictions_without_fs <- predict(mlr_model_without_fs, newdata = test_data)

# Calculate R-squared for MLR without feature selection
mean_actual_values <- mean(test_data[, target_column])
sse_without_fs <- sum((test_data[, target_column] - predictions_without_fs)^2)
sst_without_fs <- sum((test_data[, target_column] - mean_actual_values)^2)
r_squared_mlr_without_fs <- 1 - (sse_without_fs / sst_without_fs)

# Calculate RMSE for MLR without feature selection
rmse_mlr_without_fs <- sqrt(mean((test_data[, target_column] - predictions_without_fs)^2))

# Calculate MSE for MLR without feature selection
mse_mlr_without_fs <- mean((test_data[, target_column] - predictions_without_fs)^2)

# Print the metrics for Multivariate Linear Regression (MLR) without feature selection
cat("\n\nMultivariate Linear Regression (MLR) Metrics without Feature Selection:\n")
cat("R-squared:", round(r_squared_mlr_without_fs, 4), "\n")
cat("RMSE:", rmse_mlr_without_fs, "\n")
cat("MSE:", mse_mlr_without_fs, "\n")


# Multivariate Linear Regression (MLR) with Lasso Feature Selection

# Fit a Multivariate Linear Regression (MLR) model with Lasso feature selection
mlr_lasso_model <- glmnet(as.matrix(train_data[, predictor_columns]), train_data[, target_column], alpha = 1)

# Perform cross-validation to find the optimal lambda (alpha = 1 for Lasso)
cv_lasso_model <- cv.glmnet(as.matrix(train_data[, predictor_columns]), train_data[, target_column], alpha = 1)

# Get the optimal lambda for Lasso
optimal_lambda_lasso <- cv_lasso_model$lambda.min

# Fit the final Lasso model with the optimal lambda
final_lasso_model <- glmnet(as.matrix(train_data[, predictor_columns]), train_data[, target_column], alpha = 1, lambda = optimal_lambda_lasso)

# Make predictions on the test data with the Lasso-selected features
predictions_lasso <- predict(final_lasso_model, s = optimal_lambda_lasso, newx = as.matrix(test_data[, predictor_columns]))

# Calculate R-squared for Lasso-selected features
mean_actual_values <- mean(test_data[, target_column])
sse_lasso <- sum((test_data[, target_column] - as.vector(predictions_lasso))^2)
sst_lasso <- sum((test_data[, target_column] - mean_actual_values)^2)
r_squared_lasso <- 1 - (sse_lasso / sst_lasso)

# Calculate RMSE for Lasso-selected features
rmse_lasso <- sqrt(mean((test_data[, target_column] - as.vector(predictions_lasso))^2))

# Calculate MSE for Lasso-selected features
mse_lasso <- mean((test_data[, target_column] - as.vector(predictions_lasso))^2)

# Print the metrics for Multivariate Linear Regression (MLR) with Lasso feature selection
cat("\nMultivariate Linear Regression (MLR) Metrics with Lasso Feature Selection:\n")
cat("R-squared:", round(r_squared_lasso, 4), "\n")
cat("RMSE:", rmse_lasso, "\n")
cat("MSE:", mse_lasso, "\n")


