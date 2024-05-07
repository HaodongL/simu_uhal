
library(here)
library(R6)
library(hal9001)
library(glmnet)

rm(list = ls())
source(paste0(here(), "/R/uhal.R"))


# fit undersmoothed HAL
lrnr_uhal <- Lrnr_uhal9001$new()
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
uhal_fit <- lrnr_uhal$train(task)
preds <- uhal_fit$predict()
mean((preds - cpp$haz)^2)


xgb_fit <- lrnr_xgb$train(task)
xgb_preds <- xgb_fit$predict()
mean((xgb_preds - cpp$haz)^2)





