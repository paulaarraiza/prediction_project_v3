library(dplyr)
library(lubridate)
library(readr)
library(rugarch)
library(tseries)
library(zoo)
library(ggplot2)
library(stats)
library(lmtest)
library(nortest)

rm(list = ls())


generate_ts <- function(df) {  
  df <- df[c("Date", "Close")]
  df$Date <- dmy(df$Date)
  df$Close <- as.numeric(df$Close)
  
  df_ret <- df %>%
    mutate(Return = (log(Close) - lag(log(Close))))
  
  df_ret <- df_ret[-1, ]
  df_ret$Return <- df_ret$Return * 100
  
  ts_data <- zoo(df_ret$Return, order.by = df_ret$Date)
  
  all_dates_ret <- seq(min(df_ret$Date), max(df_ret$Date), by = "day")
  
  ts <- merge(ts_data, zoo(, all_dates_ret))
  ts <- na.approx(ts)
  
  # DataFrame with only Date and Return
  df_full_ret <- data.frame(Date = index(ts), Return = coredata(ts))
  
  # DataFrame with Date, Return, and Close
  df_full_close <- merge(df_full_ret, df_ret[, c("Date", "Close")], by = "Date", all.x = TRUE)
  
  # Plot the returns with ggplot2
  plot_returns <- ggplot(df_full_ret, aes(x = Date, y = Return)) +
    geom_line(color = "blue") +
    geom_hline(aes(yintercept = mean(Return)), linetype = "dashed", color = "red") +
    labs(title = "AAPL Daily Returns", x = "Date", y = "Return") +
    theme_minimal() +  
    theme(
      panel.background = element_rect(fill = "white", color = NA),  
      plot.background = element_rect(fill = "white", color = NA),   
      panel.grid.major = element_line(color = "grey90"),            
      panel.grid.minor = element_blank()  
    )
  
  return(list(df_full_ret = df_full_ret, df_full_close = df_full_close))
}

generate_log_ts <- function(df) {  
  df <- df[c("Date", "Close")]
  df$Date <- dmy(df$Date)
  df$Close <- as.numeric(df$Close)
  
  df_ret <- df %>%
    mutate(Return = log(Close) - log(lag(Close)))
  
  df_ret <- df_ret[-1, ]
  df_ret$Return <- df_ret$Return * 100
  
  ts_data <- zoo(df_ret$Return, order.by = df_ret$Date)
  
  all_dates_ret <- seq(min(df_ret$Date), max(df_ret$Date), by = "day")
  
  ts <- merge(ts_data, zoo(, all_dates_ret))
  ts <- na.approx(ts)
  
  # DataFrame with only Date and Return
  df_full_ret <- data.frame(Date = index(ts), Return = coredata(ts))
  
  # DataFrame with Date, Return, and Close
  df_full_close <- merge(df_full_ret, df_ret[, c("Date", "Close")], by = "Date", all.x = TRUE)
  
  # Plot the returns with ggplot2
  plot_returns <- ggplot(df_full_ret, aes(x = Date, y = Return)) +
    geom_line(color = "blue") +
    geom_hline(aes(yintercept = mean(Return)), linetype = "dashed", color = "red") +
    labs(title = "AAPL Daily Returns", x = "Date", y = "Return") +
    theme_minimal() +  
    theme(
      panel.background = element_rect(fill = "white", color = NA),  
      plot.background = element_rect(fill = "white", color = NA),   
      panel.grid.major = element_line(color = "grey90"),            
      panel.grid.minor = element_blank()  
    )
  
  return(list(df_full_ret = df_full_ret, df_full_close = df_full_close))
}


remove_outliers <- function(df, column_name = "Return") {
  print(paste("Removing outliers from column:", column_name, "..."))
  
  # Compute IQR
  q1 <- quantile(df[[column_name]], 0.25, na.rm = TRUE)
  q3 <- quantile(df[[column_name]], 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 2.5 * iqr
  upper_bound <- q3 + 2.5 * iqr
  
  print(paste("Outlier thresholds - Lower:", lower_bound, ", Upper:", upper_bound))
  
  # Identify outliers
  outlier_indices <- which(df[[column_name]] < lower_bound | df[[column_name]] > upper_bound)
  print(paste(length(outlier_indices), "outliers detected."))
  
  # Replace outliers with NA
  df[outlier_indices, column_name] <- NA
  
  # Interpolate missing values
  df[[column_name]] <- na.approx(df[[column_name]], na.rm = FALSE)
  
  print("Outliers removed and replaced with interpolated values.")
  return(df)
}

generate_summary_table <- function(ts, max_p, max_q, max_r, max_s) {
  
  results <- data.frame()
  
  for (p in 1:max_p) {
    for (q in 1:max_q) {
      for (r in 1:max_r) {
        for (s in 1:max_s) {
          cat("Fitting ARMA(", p, ",", q, ") + GARCH(", r, ",", s, ")...\n")
          
          tryCatch({
            spec <- ugarchspec(
              mean.model = list(armaOrder = c(p, q), include.mean = TRUE),
              variance.model = list(model = "sGARCH", garchOrder = c(r, s)),
              distribution.model = "norm"
            )
            
            garch_fit <- ugarchfit(spec, data = ts, solver = "hybrid")
            residuals <- residuals(garch_fit)
            
            # Independence Test (Ljung-Box)
            ljung_box <- Box.test(residuals, lag = 10, type = "Ljung-Box")
            independence_pval <- ljung_box$p.value
            
            # Normality Test (Lilliefors)
            lillie_test <- lillie.test(residuals)
            normality_pval <- lillie_test$p.value
            
            # Breusch-Pagan Test (Heteroskedasticity)
            bp_test <- bptest(residuals ~ seq_along(residuals))
            heteroskedasticity_pval <- bp_test$p.value
            
            # Proportion of Significant Coefficients
            coefs <- coef(garch_fit)
            se <- garch_fit@fit$matcoef[,2]  # Extract standard errors
            p_values <- 2 * (1 - pnorm(abs(coefs / se)))  # Compute p-values
            proportion_significant <- mean(p_values < 0.05)
            
            # Store results
            results <- rbind(results, data.frame(
              p = p, q = q, r = r, s = s,
              independence_pval = independence_pval,
              normality_pval = normality_pval,
              heteroskedasticity_pval = heteroskedasticity_pval,
              proportion_significant_coefficients = proportion_significant
            ))
          }, error = function(e) {
            cat("Model ARMA(", p, ",", q, ") + GARCH(", r, ",", s, ") failed with error:\n")
            cat(conditionMessage(e), "\n")  # Print the real error
          })
        }
      }
    }
  }
  
  return(results)
}


select_best_model <- function(df, alpha = 0.05) {
  # Attempt with the user-provided alpha
  filtered_df <- subset(df, 
                        independence_pval > alpha & 
                          heteroskedasticity_pval > alpha)
  
  # If no models satisfy the condition, try alpha = 0.01
  if (nrow(filtered_df) == 0) {
    warning(
      sprintf(
        "No models meet the independence and homoscedasticity criteria at alpha = %.2f. Trying alpha = 0.01.",
        alpha
      )
    )
    
    alpha <- 0.01
    filtered_df <- subset(df, 
                          independence_pval > alpha & 
                            heteroskedasticity_pval > alpha)
  }
  
  # If no models satisfy the condition, try satisfying at least one condition
  if (nrow(filtered_df) == 0) {
    warning(
      sprintf(
        "No models meet both criteria at alpha = %.2f. Trying models that satisfy at least one condition.",
        alpha
      )
    )
    
    alpha <- 0.01
    filtered_df <- subset(df, 
                          independence_pval > alpha | 
                            heteroskedasticity_pval > alpha)
  }
  
  # If still no models satisfy the condition, return NULL
  if (nrow(filtered_df) == 0) {
    warning("No models meet the independence and homoscedasticity criteria at alpha = 0.01.")
    return(NULL)
  }
  
  # Select the model(s) with the highest proportion of significant coefficients
  max_coeff <- max(filtered_df$proportion_significant_coefficients)
  best_models <- subset(filtered_df, 
                        proportion_significant_coefficients == max_coeff)
  
  # If multiple models tie, select the one with the lowest (p, q, r, s)
  best_model <- best_models[order(best_models$p, 
                                  best_models$q, 
                                  best_models$r, 
                                  best_models$s), ][1, ]
  return(list(
    best_model = best_model,
    final_alpha = alpha
  ))
}



rolling_test_prediction <- function(ts_train, ts_test, best_spec) {
  
  train_size <- length(ts_train)
  rolling_train <- ts_train
  rolling_forecasts <- numeric(length(ts_test))
  growth_forecasts <- numeric(length(ts_test))
  
  for (i in 1:length(ts_test)) {
    cat(i, "/", length(ts_test), "...\n")
    
    best_garch_fit <- ugarchfit(best_spec, data = rolling_train, solver = "hybrid")
    garch_forecast <- ugarchforecast(best_garch_fit, n.ahead = 1)
    rolling_forecasts[i] <- fitted(garch_forecast)[1]
    rolling_train <- c(rolling_train, ts_test[i])
    
    growth_forecasts[i] <- ifelse(
      rolling_forecasts[i]*ts_test[i] > 0,1, # same as realized 
      0) # otherwise
  }
  
  return(list(growth_forecasts = growth_forecasts, 
              rolling_forecasts = rolling_forecasts))
}