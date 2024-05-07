
library(here)
library(R6)
library(hal9001)
library(glmnet)
library(sl3)
library(dplyr)

rm(list = ls())
source(paste0(here(), "/R/rhal.R"))


# fit relax HAL
lrnr_rhal <- Lrnr_rhal9001$new()
lrnr_xgb <- Lrnr_xgboost$new()


# load example data set
data(cpp)
cpp <- cpp %>%
  dplyr::filter(!is.na(haz)) %>%
  mutate_all(~ replace(., is.na(.), 0))

# use covariates of intest and the outcome to build a task object
covars <- c("apgar1", "apgar5", "parity", "gagebrth", "mage", "meducyrs",
            "sexn")
task <- sl3_Task$new(
  data = cpp,
  covariates = covars,
  outcome = "haz"
)

# stack learners into a model (including screeners and pipelines)
rhal_fit <- lrnr_rhal$train(task)
preds <- rhal_fit$predict()
mean((preds - cpp$haz)^2)


xgb_fit <- lrnr_xgb$train(task)
xgb_preds <- xgb_fit$predict()
mean((xgb_preds - cpp$haz)^2)

