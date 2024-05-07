
rm(list = ls())
source(paste0(here(), "/R/gen_syn.R"))
library(foreach)
library(doParallel)

# ----------------------------------------
# TMLE ATE on original data and synthetic data
# ----------------------------------------
ate_spec <- tmle_ATE(
  treatment_level = 1,
  control_level = 0
)

# choose base learners
lrnr_mean <- make_learner(Lrnr_mean)
lrnr_xgb <- make_learner(Lrnr_xgboost)
lrnr_earth <- make_learner(Lrnr_earth)
lrnr_glm <- make_learner(Lrnr_glm)

# define metalearners appropriate to data types
ls_metalearner <- make_learner(Lrnr_nnls)
lb_metalearner <- make_learner(Lrnr_solnp,
                               learner_function = metalearner_logistic_binomial,
                               loss_function = loss_loglik_binomial)
sl_Y <- Lrnr_sl$new(
  learners = list(lrnr_mean, lrnr_glm, lrnr_earth),
  metalearner = ls_metalearner
)
sl_A <- Lrnr_sl$new(
  learners = list(lrnr_mean, lrnr_glm, lrnr_earth),
  metalearner = lb_metalearner
)
learner_list <- list(A = sl_A, Y = sl_Y)



B <- 500
# num_cores <- 2
# registerDoParallel(cores = num_cores)

df_res <- foreach(b = 1:B) %do% {
  cat(" b = ",b,"\n")
  set.seed(123 + b)
  # ----------------------------------------
  # Generate Data 
  # ----------------------------------------
  # generate data from truth
  df_true <- generate_data_simple(500)
  
  # generate syn data from uhal fit
  df_syn_uhal <- generate_data(n = nrow(df),
                          df = df,
                          y_type = "continuous",
                          g_fit = res_Qg_uhal$g_fit,
                          Q_fit = res_Qg_uhal$Q_fit)
  
  # generate syn data from uhal fit
  df_syn_rhal <- generate_data(n = nrow(df),
                          df = df,
                          y_type = "continuous",
                          g_fit = res_Qg_rhal$g_fit,
                          Q_fit = res_Qg_rhal$Q_fit)
  
  # generate syn data from sl fit
  df_syn_sl <- generate_data(n = nrow(df),
                          df = df,
                          y_type = "continuous",
                          g_fit = res_Qg_sl$g_fit,
                          Q_fit = res_Qg_sl$Q_fit)
  
  # generate syn data with synthpop
  suppressWarnings(
  df_syn_pop <- syn(df)$syn
  )
  
  # generate syn data with synthpop + uhal
  # simu_data0 <- df_syn_pop %>% mutate(A = df$A, Y = df$Y)
  # df_syn_popuhal <- generate_data(n = nrow(df),
  #                              df = df,
  #                              y_type = "continuous",
  #                              g_fit = res_Qg_uhal$g_fit,
  #                              Q_fit = res_Qg_uhal$Q_fit,
  #                              simu_data0 = simu_data0)
  
  # generate syn data with rgan
  suppressWarnings(
    df_syn_gan <- sample_synthetic_data(trained_gan, transformer)
  )
  df_syn_gan <- data.frame(df_syn_gan)
  
  # ----------------------------------------
  # Calc Estimates
  # ----------------------------------------
  # Get the empirical estimates of simple parameters
  simple_true <- get_emp(df_true)
  simple_sl <- get_emp(df_syn_sl)
  simple_rhal <- get_emp(df_syn_rhal)
  simple_uhal <- get_emp(df_syn_uhal)
  simple_pop <- get_emp(df_syn_pop)
  # simple_popuhal <- get_emp(df_syn_popuhal)
  simple_gan <- get_emp(df_syn_gan)
  
  df_res_simple <- rbind(cbind(simple_true, "type" = "true"),
                         cbind(simple_sl, "type" = "SL"),
                         cbind(simple_rhal, "type" = "r-HAL"),
                         cbind(simple_uhal, "type" = "u-HAL"),
                         cbind(simple_pop, "type" = "synthpop"),
                         cbind(simple_gan, "type" = "RGAN"))
  
  
  # Get TMLE estimatws of ATE
  tmle_fit_true <- tmle3(ate_spec, df_true, node_list, learner_list)
  tmle_fit_sl <- tmle3(ate_spec, df_syn_sl, node_list, learner_list)
  tmle_fit_rhal <- tmle3(ate_spec, df_syn_rhal, node_list, learner_list)
  tmle_fit_uhal <- tmle3(ate_spec, df_syn_uhal, node_list, learner_list)
  tmle_fit_pop <- tmle3(ate_spec, df_syn_pop, node_list, learner_list)
  # tmle_fit_popuhal <- tmle3(ate_spec, df_syn_popuhal, node_list, learner_list)
  tmle_fit_gan <- tmle3(ate_spec, df_syn_gan, node_list, learner_list)
  
  df_res_tmle <- rbind(cbind("est" = tmle_fit_true$summary$tmle_est,
                             "se" = tmle_fit_true$summary$se, 
                             "type" = "true"),
                       cbind("est" = tmle_fit_sl$summary$tmle_est,
                             "se" = tmle_fit_sl$summary$se, 
                             "type" = "SL"),
                       cbind("est" = tmle_fit_rhal$summary$tmle_est,
                             "se" = tmle_fit_rhal$summary$se, 
                             "type" = "r-HAL"),
                       cbind("est" = tmle_fit_uhal$summary$tmle_est,
                             "se" = tmle_fit_uhal$summary$se, 
                             "type" = "u-HAL"),
                       cbind("est" = tmle_fit_pop$summary$tmle_est,
                             "se" = tmle_fit_pop$summary$se, 
                             "type" = "synthpop"),
                       cbind("est" = tmle_fit_gan$summary$tmle_est,
                             "se" = tmle_fit_gan$summary$se, 
                             "type" = "RGAN")
                       ) %>% as.data.frame()
  
  df_res_i <- df_res_simple %>% left_join(df_res_tmle, by = "type")
  
  df_res_i
}

df_res_all <- rbindlist(df_res)
df_res_all$est <- as.numeric(df_res_all$est)
df_res_all$se <- as.numeric(df_res_all$se)

# saveRDS(df_res_all, paste0(here(), "/simu/df_res_all_500.RDS"))
df_res_all <- readRDS(paste0(here(), "/simu/df_res_all_1000.RDS"))

calc_perf <- function(x,tru,se = NULL) {
  bias <- (mean(x)-tru)
  varx <- var(x)
  MSE <- bias^2+varx
  
  if (is.null(se)){
    coverage <- NA
  }else{
    ci_l <- x - 1.96*se
    ci_u <- x + 1.96*se
    coverage <- mean((ci_l < tru) * (ci_u > tru))
  }
  df_perf <- data.frame("bias" = bias,
                        "varx" = varx,
                        "MSE" = MSE,
                        "coverage" = coverage)
  return(df_perf)
}

get_summary <- function(df_res){
  df_truth <- get_truth()
  df_summary <- data.frame()
  for (t in c("true", "SL", "r-HAL", "u-HAL", "synthpop", "RGAN")){
    df_res_sub <- df_res %>% filter(type == t)
    sum_EAn <- calc_perf(df_res_sub$EAn, df_truth$P0A)
    sum_EYn <- calc_perf(df_res_sub$EYn, df_truth$P0Y)
    sum_BetaA <- calc_perf(df_res_sub$BetaA, df_truth[[4]][2], df_res_sub$BetaAse)
    sum_ATE <- calc_perf(df_res_sub$est, df_truth$true.ATE, df_res_sub$se)
    
    df_sum_t <- bind_rows(c(sum_EAn, "para" = "EA", "type" = t),
                      c(sum_EYn, "para" = "EY", "type" = t),
                      c(sum_BetaA, "para" = "BetaA", "type" = t),
                      c(sum_ATE, "para" = "ATE", "type" = t))
    
    df_summary <- bind_rows(df_summary, df_sum_t)
  }
  return(df_summary)
}


df_summary <- get_summary(df_res_all)

library(tidyr)
library(ggplot2)

df_summary_long <- df_summary %>% 
  mutate(bias2 = bias^2) %>% 
  gather(metric, amount, c(bias2, varx))

# df_summary %>%  ggplot(aes(x = type, y = bias)) +  
#   geom_bar(stat = "identity") + 
#   facet_wrap(~para)

df_summary_long %>% filter(type != "RGAN") %>%  
  ggplot(aes(x = type, y = amount, fill = metric)) +  
  geom_bar(stat = "identity") + 
  facet_wrap(~para)

df_summary_long %>% filter(para == "ATE", type != "RGAN") %>%  
  ggplot(aes(x = type, y = amount, fill = metric)) +  
  geom_bar(stat = "identity")
ggsave(paste0(here(), "/plots/p_ate.png"))

df_summary_long %>% filter(para == "EA", type != "RGAN") %>%  
  ggplot(aes(x = type, y = amount, fill = metric)) +  
  geom_bar(stat = "identity")
ggsave(paste0(here(), "/plots/p_EA.png"))

df_summary_long %>% filter(para == "EY", type != "RGAN") %>%  
  ggplot(aes(x = type, y = amount, fill = metric)) +  
  geom_bar(stat = "identity")
ggsave(paste0(here(), "/plots/p_EY.png"))

df_summary_long %>% filter(para == "BetaA", type != "RGAN") %>%  
  ggplot(aes(x = type, y = amount, fill = metric)) +  
  geom_bar(stat = "identity")
ggsave(paste0(here(), "/plots/p_BetaA.png"))
