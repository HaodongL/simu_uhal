


library(data.table)
library(dplyr)
library(glmnet)
library(R6)
library(sl3)
library(tmle3)
library(hal9001)

rm(list = ls())
source(".../uhal.R")


# ----------------------------------------
# Generate Original Data 
# ----------------------------------------
generate_data_simple <- function(N){
  # A simple data generating process
  X1 <- runif(N,-1,1)
  X2 <- runif(N,-1,1)
  pi0 <- plogis(0.1*X1*X2-0.4*X1)
  A <- rbinom(N,1,prob=pi0)
  muY0 <- X1*X2 + 2*X2^2 -X1
  CATE <- X1^2*(X1+7/5) + (5*X2/3)^2
  muY = muY0+A*CATE
  Y <- rnorm(N,sd=1,mean= muY)
  return(tibble(X1=X1,X2=X2,A=A,Y=Y))
}

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

lrnr_uhal <- Lrnr_uhal9001$new()

res_Qg <- fit_uhal_Qg(df = df, 
                      y_type = "continuous", 
                      covars = node_list$W)

set.seed(123)
df_syn <- generate_uhal_data(n = nrow(df), 
                             df = df, 
                             y_type = "continuous", 
                             g_fit = res_Qg$g_fit, 
                             Q_fit = res_Qg$Q_fit)


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


# estimate ATE with original data
tmle_fit_org <- tmle3(ate_spec, df, node_list, learner_list)
print(tmle_fit_org)


# estimate ATE with synthetic data
tmle_fit_syn <- tmle3(ate_spec, df_syn, node_list, learner_list)
print(tmle_fit_syn)

