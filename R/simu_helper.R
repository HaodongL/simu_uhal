
# helper functions for simulation with uhal or rhal


generate_data <- function(n = 500, df, y_type, g_fit, Q_fit){
  
  #--- generate W ---#
  covars <- setdiff(names(df), c("A","Y"))
  # simulate W from the empirical distribution of W in the real data
  simu_data <- dplyr::sample_n(df, size = n, replace = TRUE)
  
  #--- generate A ---#
  task_a_predict <- make_sl3_Task(data = simu_data, covariates = covars,
                                  outcome = "A", outcome_type = 'binomial')
  a_preds <- g_fit$predict(task_a_predict)
  simu_data$A <- rbinom(n, 1, prob = a_preds)
  
  #--- generate Y ---#
  task_y_predict <- make_sl3_Task(data = simu_data, covariates = c(covars, "A"),
                                  outcome = "Y", outcome_type = y_type)
  if (y_type == "continuous"){
    y_preds <- Q_fit$predict(task_y_predict)
    rv <- sum((y_preds - simu_data$Y)^2)/n
    y_preds_error <- rnorm(n, mean = 0, sd = sqrt(rv))
    y_preds <- y_preds + y_preds_error
    simu_data$Y <- y_preds
  }else{
    simu_data$Y <- rbinom(n, 1, prob = y_preds)
  }
  
  # # drop constant cols
  # for (covs in setdiff(names(simu_data), c('Y', 'A'))) {
  #   if (length(unique(simu_data[[covs]])) == 1) {
  #     simu_data = simu_data %>% select(-all_of(covs))
  #   }
  # }
  
  return(simu_data)
}


fit_sl_Qg <- function(df, sl_Q, sl_g, y_type, covars){
  task_Q <- sl3_Task$new(
    data = df,
    covariates = c(covars, "A"),
    outcome = "Y",
    outcome_type = y_type
  )
  Q_fit <- sl_Q$train(task_Q)
  
  task_g <- sl3_Task$new(
    data = df,
    covariates = covars,
    outcome = "A",
    outcome_type = "binomial"
  )
  g_fit <- sl_g$train(task_g)
  
  res <- list("Q_fit" = Q_fit,
              "g_fit" = g_fit)
  return(res)
}