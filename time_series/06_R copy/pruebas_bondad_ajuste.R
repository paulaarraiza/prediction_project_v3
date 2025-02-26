# Load necessary libraries
library(fitdistrplus)
library(MASS)
library(nortest)
library(actuar)  # For more distributions like Weibull, Burr, etc.
library(ggplot2)

# Example: Replace with your actual data
set.seed(123)
data <- rnorm(1000, mean = 5, sd = 2)  # Example normal data

# 1. **Visual Inspection**
par(mfrow = c(2,2))
hist(data, breaks = 30, probability = TRUE, col = "lightblue", main = "Histogram with Density")
lines(density(data), col = "red", lwd = 2)

qqnorm(data)
qqline(data, col = "red", lwd = 2)

# 2. **Fit Distributions**
fits <- list(
  normal = fitdist(data, "norm"),
  gamma = fitdist(data, "gamma"),
  lognormal = fitdist(data, "lnorm"),
  weibull = fitdist(data, "weibull"),
  exponential = fitdist(data, "exp"),
  beta = tryCatch(fitdist(data, "beta"), error = function(e) NULL)  # Beta may fail for some data
)

# 3. **Compare Distributions using AIC/BIC**
fit_results <- data.frame(
  Distribution = names(fits),
  AIC = sapply(fits, function(f) if (!is.null(f)) f$aic else NA),
  BIC = sapply(fits, function(f) if (!is.null(f)) BIC(f) else NA)
)

fit_results <- fit_results[order(fit_results$AIC), ]
print(fit_results)

# 4. **Goodness-of-Fit Tests**
ks_tests <- sapply(fits, function(f) {
  if (!is.null(f)) {
    distname <- f$distname
    params <- as.list(f$estimate)  # Extract parameters dynamically
    
    # Apply the KS test correctly for each distribution
    if (distname == "norm") {
      ks.test(data_shifted, "pnorm", mean = params$mean, sd = params$sd)$p.value
    } else if (distname == "gamma") {
      ks.test(data_shifted, "pgamma", shape = params$shape, rate = params$rate)$p.value
    } else if (distname == "lnorm") {
      ks.test(data_shifted, "plnorm", meanlog = params$meanlog, sdlog = params$sdlog)$p.value
    } else if (distname == "weibull") {
      ks.test(data_shifted, "pweibull", shape = params$shape, scale = params$scale)$p.value
    } else if (distname == "exp") {
      ks.test(data_shifted, "pexp", rate = params$rate)$p.value
    } else if (distname == "beta") {
      ks.test(data_shifted, "pbeta", shape1 = params$shape1, shape2 = params$shape2)$p.value
    } else {
      NA  # Return NA for unsupported distributions
    }
  } else {
    NA  # Handle cases where fitting failed
  }
})


ad_tests <- sapply(fits, function(f) {
  if (!is.null(f)) ad.test(data)$p.value else NA
})

test_results <- data.frame(
  Distribution = names(fits),
  KS_p_value = ks_tests,
  AD_p_value = ad_tests
)
print(test_results)

# 5. **Best Distribution Selection**
best_dist <- fit_results$Distribution[1]
cat("The best-fitting distribution based on AIC is:", best_dist, "\n")

