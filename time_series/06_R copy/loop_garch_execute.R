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
library(forecast)

rm(list = ls())

source("/Users/paulaarraizaarias/Documents/ucm/tfg/matematicas/06_R/log_garch_modeling.R")

data_path <- "/Users/paulaarraizaarias/Documents/ucm/tfg/matematicas/07_data"
docs_path <- "/Users/paulaarraizaarias/Documents/ucm/tfg/matematicas/05_docs"
results_path <- "/Users/paulaarraizaarias/Documents/ucm/tfg/matematicas/08_results"

stock_names <- c("AAPL", "AMZN", "BTC", "GOOGL", "META", 
                 "MSFT", "NVDA", "SPX", "WTI")

perc_train_list <- c(0.995)
start_dates <- c("2015-01-01", "2020-01-01", "2022-01-01", "2023-01-01", "2024-01-01")
# start_dates <- c("2009-01-01")

for (filter_start_date in start_dates) {
  for (train_perc in perc_train_list) {
    for (stock_name in stock_names){
      
      cat(stock_name, train_perc, filter_start_date)
      
      if (stock_name %in% c("BTC", "GOOGL", "META", "WTI")) {
        data_raw <- read.csv(file.path(data_path, paste0("stocks/", stock_name, "_Close.csv")), sep=",", dec=".",stringsAsFactors=FALSE) 
      } else {
        data_raw <- read.csv(file.path(data_path, paste0("stocks/", stock_name, "_Close.csv")), sep=";", dec=",",stringsAsFactors=FALSE) 
      }
      
      result_df_raw <- generate_log_ts(data_raw)$df_full_ret
      result_df_raw <- remove_outliers(result_df_raw)
      if (any(is.na(result_df_raw))) {
        cat("Removed ", sum(is.na(result_df_raw)))
        result_df_raw <- na.omit(result_df_raw)
      }
      
      max_p <- 3
      max_q <- 3  
      max_r <- 3  
      max_s <- 3  
      
      # Filter by dates if needed
      # filter_start_date <- "2015-01-01"
      filter_end_date <- "2024-12-31"
      
      result_df <- result_df_raw %>%
        filter(Date >= as.Date(filter_start_date), 
               Date <= as.Date(filter_end_date))
      
      start_date <- result_df[1, 1]
      end_date <- result_df[nrow(result_df), 1]
      
      ts <- result_df$Return
      n <- length(ts)
      
      perc_str <- as.character(train_perc * 100)
      formatted_perc <- gsub("\\.", "_",perc_str)
      
      train_size <- floor(train_perc * n)
      # train_size <- 5300
      
      ts_train <- ts[1:train_size]
      ts_test <- ts[(train_size+1):n]
      
      alpha <- 0.05
      
      start_time <- Sys.time()
      
      results <- generate_summary_table(ts_train, max_p, max_q, max_r, max_s)
      
      best_model_choice <- select_best_model(results, alpha)
      best_model <- best_model_choice$best_model
      
      best_p <- best_model$p
      best_q <- best_model$q
      best_r <- best_model$r
      best_s <- best_model$s
      
      alpha <- best_model_choice$final_alpha
      independence_pval <- best_model$independence_pval
      heteroskedasticity_pval <- best_model$heteroskedasticity_pval
      normality_pval <- best_model$normality_pval
      prop_significant_coef <- best_model$proportion_significant_coefficients
      
      best_spec <- ugarchspec(
        mean.model = list(armaOrder = c(best_p, best_q), include.mean = TRUE),
        variance.model = list(model = "sGARCH", garchOrder = c(best_r, best_s)),
        distribution.model = "norm"
      )
      print(results)
      
      if (!(best_p == 0 & best_q == 0 & best_r == 0 & best_s == 0)) {
        prediction_results <- rolling_test_prediction(ts_train, ts_test, best_spec)
        growth_forecasts <- prediction_results$growth_forecasts
        rolling_forecasts <- prediction_results$rolling_forecasts
        accuracy <- mean(growth_forecasts)
        
      } else {
        accuracy <- 0
      }
      end_time <- Sys.time()
      execution_seconds <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      result <- data.frame(
        stock_name = stock_name,
        start_date = as.Date(start_date),
        end_date = as.Date(end_date),
        train_size = train_size,
        test_size = (n-train_size),
        train_perc =train_perc*100,
        seconds = execution_seconds,
        accuracy = accuracy,
        model_params = paste0("(", best_p, ",", best_q, ",", best_r, ",", best_s, ")"),
        alpha = alpha,
        independence_pval = independence_pval,
        heteroskedasticity_pval = heteroskedasticity_pval,
        normality_pval = normality_pval,
        prop_significant_coef= prop_significant_coef,
        execution_date = Sys.Date()
      )
      
      years_diff <- round(as.numeric(difftime(end_date, start_date, units = "days")) / 365.25, 0)
      
      cat("Years: ", years_diff, "stock: ", stock_name, "Perc: ", formatted_perc)
      
      output_filepath <- file.path(results_path, paste0("log_time_series/results_", years_diff, "y_", formatted_perc, ".csv"))
      
      if (!file.exists(output_filepath)) {
        write.csv(result, output_filepath, row.names = FALSE)
      } else {
        existing_data <- read.csv(output_filepath)
        
        existing_data$start_date <- as.Date(existing_data$start_date)
        existing_data$end_date <- as.Date(existing_data$end_date)
        existing_data$execution_date <- as.Date(existing_data$execution_date)
        
        updated_data <- bind_rows(existing_data, result)
        write.csv(updated_data, output_filepath, row.names = FALSE)
      }
    }
  }
}    
