# load libraries
.libPaths(c(.libPaths(), "/snfs1/temp/scj13/r_packages/3.5.1_fast"))
library(data.table)
library(boot)
library(arm)
library(tidyr)
library(merTools)
library(ggplot2)
library(dplyr)
library(lme4)
library(lmerTest)
library(caret)

args <- commandArgs(trailingOnly = T)
int_cause <- as.character(args[1])
data_dir <- as.character(args[2])
diag_dir <- as.character(args[3])
obs_fraction_col <- as.character(args[4])

source("/ihme/cc_resources/libraries/current/r/get_location_metadata.R")

get_covariates <- function() {
  covariates_df <- fread(paste0(Sys.getenv("HOME"), "/cod-data/mcod_prep/maps/covariates.csv"))
  int_cause_str <- int_cause
  covariates <- covariates_df[covariates_df$int_cause == int_cause_str, covariates]
  covariates <- unlist(strsplit(covariates, ", "))
  return(covariates)
}

get_formula <- function(covariates) {
  base_covariates <- c("sex_id", "age_group_id", "(1|level_1/level_2)")
  exp_vars <- c(covariates, base_covariates)
  resp_vars <- c("successes", "failures")
  formula <- reformulate(exp_vars, parse(text = sprintf("cbind(%s)", toString(resp_vars)))[[1]])
  return(formula)
}

run_model <- function(data, covariates) {
  print(paste0(Sys.time(), " Running mixed effect binomial model for ", int_cause))
  formula <- get_formula(covariates)
  model <- glmer(
    formula = formula, data = data, family = binomial(link = "logit"), 
    nAGQ = 1, verbose = 0, control = glmerControl(optimizer = "Nelder_Mead", 
      optCtrl = list(maxfun = 2e5))
  )
  converged <- check_model_converged(model)
  if (!converged) {
    print("Model failed to converge! Trying bobyqa.")
    model <- update(model, control = glmerControl(optimizer = "bobyqa"))
    print(paste0(Sys.time(), summary(model)))
    converged <- check_model_converged(model)
    if (!converged) {
      print("Model did not converge with bobyqa! Trying again with nloptwrap")
      model <- update(model, control = glmerControl(optimizer = "nloptwrap"))
      print(paste0(Sys.time(), summary(model)))
      converged <- check_model_converged(model)
      if (!converged) {
        print("Model failed to converge with all 3 optimizers, need to troubleshoot.")
      }
    }
  }
  print(paste0(Sys.time(), " Saving model output"))
  saveRDS(model, paste0(data_dir, "/", int_cause, "_model.rds"))
  return(model)
}

check_model_converged <- function(model) {
  converged <- FALSE
  # depending on the optimizer, convergence codes of 0/1 can be either convergence
  # or non-convergence. additionally, some warning messages are reportedly inaccurate
  # (https://www.r-project.org/nosvn/pandoc/lme4.html)
  # per lme4 documentation, use these checks to see if everything is good:
  # 1) check singularity
  diag.vals <- getME(model, "theta")[getME(model, "lower") == 0]
  # want this to be false
  if (!(any(diag.vals < 1e-6))) {
    check1 <- TRUE
  }

  # 2) recompute gradient and Hessian with Richardson extrapolation
  devfun <- update(model, devFunOnly = TRUE)
  if (isLMM(model)) {
    pars <- getME(model, "theta")
  } else {
    # GLMM: requires both random and fixed parameters
    pars <- getME(model, c("theta", "fixef"))
  }
  if (require("numDeriv")) {
    cat("hess:\n")
    print(hess <- hessian(devfun, unlist(pars)))
    cat("grad:\n")
    print(grad <- grad(devfun, unlist(pars)))
    cat("scaled gradient:\n")
    print(scgrad <- solve(chol(hess), grad))
  }
  # compare with internal calculations
  # not sure what threshold of a comparison this should be
  dd <- model@optinfo$derivs

  # want this to be true
  if (with(dd, max(abs(solve(Hessian, gradient))) < 2e-3)) {
    check2 <- TRUE
  } else{
    check2 <- FALSE
  }

  if (check2 & check1) {
    converged <- TRUE
  }

  return(converged)
}

get_rmse <- function(df) {
  df[, residual_squared := (preds - get(obs_fraction_col))^2]
  rmse <- sqrt(mean(df$residual_squared))
  return(rmse)
}

save_diagnostics <- function(model, train, test, covariates) {
  print(paste0(Sys.time(), " Saving diagnostics"))
  # predict values for training data and save as a diagnostic compared to training values
  preds <- predict(model, test)
  test[, preds_logit := (preds)]
  test[, preds := inv.logit(preds)]
  write.csv(test, paste0(data_dir, "/", int_cause, "_model_diagnostics.csv"), row.names = F)

  # save coefficients
  adjustments <- list()
  beta <- fixef(model)
  beta_se <- se.fixef(model)
  adjustments[[length(adjustments) + 1]] <- data.table(beta = beta, beta_se = beta_se, cv = names(fixef(model)), rmse = get_rmse(test))
  adjustments <- rbindlist(adjustments)
  # report in odds, so take the exp of the log odds (logit)
  adjustments[cv == "(Intercept)", beta := inv.logit(beta)]
  adjustments[cv == "(Intercept)", beta_se := inv.logit(beta_se)]
  adjustments[cv != "(Intercept)", exp_beta := exp(beta)]
  adjustments[cv != "(Intercept)", lower := exp(beta - beta_se * 1.96)]
  adjustments[cv != "(Intercept)", upper := exp(beta + beta_se * 1.96)]
  write.csv(adjustments, paste0(diag_dir, "/", int_cause, "_model_coefficients.csv"), row.names = F)

  # get country column
  loc_df <- get_location_metadata(location_set_version_id=420)[, c("location_id", "ihme_loc_id")]
  test <- merge(loc_df, test, by=c("location_id"))
  train <- merge(loc_df, train, by=c("location_id"))
  test$iso3 <- substr(test$ihme_loc_id, 1, 3)
  train$iso3 <- substr(train$ihme_loc_id, 1, 3)

  # plots!
  for (covariate in covariates) {
    # histogram of covariates
    ggplot(train, aes_string(x=covariate, color="iso3")) + geom_density()
    ggsave(paste0(diag_dir, "/hist_", covariate, ".pdf"), width = 11, height = 8.5)
  }

    xlab <- paste0("Observed, ", obs_fraction_col)
    ylab <- paste0("Prediction, ", obs_fraction_col)
    # plot scatter of observed & predicted
    ggplot(test) + geom_point(aes_string(x=obs_fraction_col, y="preds", color='iso3'), size=3, alpha=0.4) +
    labs(color="Country") + theme_bw() + ylab(ylab) + xlab(xlab) +
    geom_abline(intercept=0, slope=1, color='grey') +
    theme(legend.position="bottom") + ggtitle("Predicted vs. observed values")
    ggsave(paste0(diag_dir, "/preds_obs", ".pdf"), width = 11, height = 8.5)
  
  # get residuals
  test[, residuals := (preds_logit - get(logit_obs_fraction_col))]

# plot residuals for each covariate
for (covariate in covariates) {
    ggplot(test) + geom_point(aes_string(x=covariate, y="residuals", color="iso3"), size=3, alpha=0.4) +
      labs(color="Country") +
      theme_bw() + ylab("Residuals") + xlab(covariate) +
      geom_abline(intercept=0, slope=0, color='grey') +
      theme(legend.position="bottom") + ggtitle("Residuals")
    ggsave(paste0(diag_dir, "/residuals_", covariate, ".pdf"), width = 11, height = 8.5)

  }
}

# read in all cause data
data <- fread(paste0(data_dir, "/model_input.csv"))

# read in covariates (in addition to age, sex, cause)
covariates <- get_covariates()

# create categorical variables
data$sex_id <- factor(data$sex_id)
data$age_group_id <- factor(data$age_group_id)
data$level_1 <- factor(data$level_1)
data$level_2 <- factor(data$level_2)

# create test/train datasets
set.seed(52)
trainIndex <- createDataPartition(data$level_2, p=0.75, list=FALSE)
train = data[trainIndex,]
test = data[-trainIndex,]

model <- run_model(train, covariates)

save_diagnostics(model, train, test, covariates)