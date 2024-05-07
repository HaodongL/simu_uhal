library(data.table)
library(dplyr)
library(glmnet)
library(R6)
library(sl3)
library(tmle3)
library(hal9001)
library(synthpop)
library(facefuns)
library(RGAN)


rm(list = ls())
source(paste0(here(), "/R/uhal.R"))
source(paste0(here(), "/R/rhal.R"))
source(paste0(here(), "/R/simu_helper.R"))

# ----------------------------------------
# Helper functions for generating syn data
# ----------------------------------------

## Generate data from known DGD
generate_data_simple <- function(N,Xranges=c(-1,1,-1,1),betaA=c(0,0.1,-0.4),
                                 betaY0=c(0,1,2,-1),betaC=c(1,7/5,5,3),sdy=1){
  # A simple data generating process
  X1 <- runif(N,Xranges[1],Xranges[2])
  X2 <- runif(N,Xranges[3],Xranges[4])
  pi0 <- plogis(betaA[1]+betaA[2]*X1*X2+betaA[3]*X1)
  A <- rbinom(N,1,prob=pi0)
  muY0 <- betaY0[1]+betaY0[2]*X1*X2 + betaY0[3]*X2^2 +betaY0[4]*X1
  CATE <- betaC[1]*X1^2*(X1+betaC[2]) + (betaC[3]*X2/betaC[4])^2
  muY = muY0+A*CATE
  Y <- rnorm(N,sd=sdy,mean= muY)
  return(tibble(X1=X1,X2=X2,A=A,Y=Y))
}


## Get True_values from known DGD
get_truth <- function(N=100000,Xranges=c(-1,1,-1,1),
                      betaA=c(0,0.1,-0.4),                                   
                      betaY0=c(0,1,2,-1),
                      betaC=c(1,7/5,5,3),sdy=1){
  # A simple data generating process
  X1 <- runif(N,Xranges[1],Xranges[2])
  X2 <- runif(N,Xranges[3],Xranges[4])
  pi0 <- plogis(betaA[1]+betaA[2]*X1*X2+betaA[3]*X1)
  P0A = mean(pi0)
  A <- rbinom(N,1,prob=pi0)
  muY0 <- betaY0[1]+betaY0[2]*X1*X2 + betaY0[3]*X2^2 +betaY0[4]*X1
  CATE <- betaC[1]*X1^2*(X1+betaC[2]) + (betaC[3]*X2/betaC[4])^2
  true.ATE <- mean(CATE)
  muY = muY0+A*CATE
  Y <- rnorm(N,sd=sdy,mean= muY)
  P0Y <- mean(Y)
  coef.work <- coef(lm(Y~A+X1+X2))
  return(list(P0A=P0A,P0Y=P0Y,true.ATE=true.ATE,coef.work))
}

# Get estimates of simple parameters
get_emp <- function(df){
  aveAn <- mean(df$A)
  aveYn <- mean(df$Y)
  lmn <- lm(Y~A+X1+X2,data=df)
  coeffn <- summary(lmn)$coefficients[,1:2]
  return(data.frame(EAn = aveAn,
              EYn = aveYn,
              BetaA = coeffn[,1][2],
              BetaAse = coeffn[,2][2]))
}




# ----------------------------------------
# Generate Original Data 
# ----------------------------------------
# generate_data_simple <- function(N){
#   # A simple data generating process
#   X1 <- runif(N,-1,1)
#   X2 <- runif(N,-1,1)
#   pi0 <- plogis(0.1*X1*X2-0.4*X1)
#   A <- rbinom(N,1,prob=pi0)
#   muY0 <- X1*X2 + 2*X2^2 -X1
#   CATE <- X1^2*(X1+7/5) + (5*X2/3)^2
#   muY = muY0+A*CATE
#   Y <- rnorm(N,sd=1,mean= muY)
#   return(tibble(X1=X1,X2=X2,A=A,Y=Y))
# }

# true ATE = 1.39
set.seed(123)
df <- generate_data_simple(500)



# ----------------------------------------
# Generate Synthetic Data with undersmoothed HAL
# ----------------------------------------
node_list <- list(
  W = setdiff(names(df), c("A", "Y")),
  A = "A",
  Y = "Y"
)

sl_Q_uhal <- Lrnr_uhal9001$new()
sl_g_uhal <- Lrnr_uhal9001$new()

res_Qg_uhal <- fit_Qg(df = df, 
                 sl_Q = sl_Q_uhal,
                 sl_g = sl_g_uhal,
                 y_type = "continuous", 
                 covars = node_list$W)


# set.seed(123)
# df_syn_uhal <- generate_data(n = nrow(df), 
#                         df = df, 
#                         y_type = "continuous", 
#                         g_fit = res_Qg_uhal$g_fit, 
#                         Q_fit = res_Qg_uhal$Q_fit)


# ----------------------------------------
# Generate Synthetic Data with relaxed HAL
# ----------------------------------------
node_list <- list(
  W = setdiff(names(df), c("A", "Y")),
  A = "A",
  Y = "Y"
)

sl_Q_rhal <- Lrnr_rhal9001$new()
sl_g_rhal <- Lrnr_rhal9001$new()

res_Qg_rhal <- fit_Qg(df = df, 
                 sl_Q = sl_Q_rhal,
                 sl_g = sl_g_rhal,
                 y_type = "continuous", 
                 covars = node_list$W)


# set.seed(123)
# df_syn_rhal <- generate_data(n = nrow(df), 
#                         df = df, 
#                         y_type = "continuous", 
#                         g_fit = res_Qg_rhal$g_fit, 
#                         Q_fit = res_Qg_rhal$Q_fit)


# ----------------------------------------
# Generate Synthetic Data with Super Learner
# ----------------------------------------
node_list <- list(
  W = setdiff(names(df), c("A", "Y")),
  A = "A",
  Y = "Y"
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
sl_Q <- Lrnr_sl$new(
  learners = list(lrnr_mean, lrnr_glm, lrnr_earth, lrnr_xgb),
  metalearner = ls_metalearner
)
sl_g <- Lrnr_sl$new(
  learners = list(lrnr_mean, lrnr_glm, lrnr_earth, lrnr_xgb),
  metalearner = lb_metalearner
)

res_Qg_sl <- fit_Qg(df = df, 
                 sl_Q = sl_Q,
                 sl_g = sl_g,
                 y_type = "continuous", 
                 covars = node_list$W)


# set.seed(123)
# df_syn_sl <- generate_data(n = nrow(df), 
#                         df = df, 
#                         y_type = "continuous", 
#                         g_fit = res_Qg_sl$g_fit, 
#                         Q_fit = res_Qg_sl$Q_fit)


# ----------------------------------------
# Generate Synthetic Data with synthpop
# ----------------------------------------
# df_syn_pop <- syn(df)$syn


# ----------------------------------------
# Generate Synthetic Data with RGAN
# ----------------------------------------
# torch::install_torch() # need to install torch for the first time use

# Build new transformer
transformer <- data_transformer$new()
# Fit transformer to data
transformer$fit(as.matrix(df))
# Transform data and store as new object
transformed_data <- transformer$transform(as.matrix(df))
# Train the default GAN
trained_gan <- gan_trainer(transformed_data)
# Sample synthetic data from the trained GAN
# df_syn_gan <- sample_synthetic_data(trained_gan, transformer)
# df_syn_gan <- data.frame(df_syn_gan)

# Plot the results
# GAN_update_plot(data = as.matrix(df),
#                 synth_data = df_syn_gan,
#                 main = "Real and Synthetic Data after Training")

# # ----------------------------------------
# # Compare W via split violin plots
# # ----------------------------------------
# df_all <- rbind(df %>% mutate(method = "synthpop", cond = "Truth"),
#                 df %>% mutate(method = "u-HAL", cond = "Truth"),
#                 df %>% mutate(method = "r-HAL", cond = "Truth"),
#                 df %>% mutate(method = "SL", cond = "Truth"),
#                 df %>% mutate(method = "RGAN", cond = "Truth"),
#                 df_syn_pop %>% mutate(method = "synthpop", cond = "Synthetic"),
#                 df_syn_uhal %>% mutate(method = "u-HAL", cond = "Synthetic"),
#                 df_syn_rhal %>% mutate(method = "r-HAL", cond = "Synthetic"),
#                 df_syn_sl %>% mutate(method = "SL", cond = "Synthetic"),
#                 df_syn_gan %>% mutate(method = "RGAN", cond = "Synthetic"))
# 
# 
# 
# ggplot2::ggplot(df_all, ggplot2::aes(method, A, fill=cond)) +
#   geom_split_violin()
# 
# 
# ggplot2::ggplot(df_all, ggplot2::aes(method, Y, fill=cond)) +
#   geom_split_violin()
# 
# 
# ggplot2::ggplot(df_all, ggplot2::aes(method, X1, fill=cond)) +
#   geom_split_violin()


