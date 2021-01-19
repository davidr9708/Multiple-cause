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
library(betareg)
library(glmnet)
library(Matrix)

args <- commandArgs(trailingOnly = T)
int_cause <- as.character(args[1])
data_dir <- as.character(args[2])
diag_dir <- as.character(args[3])
successes <- paste0(int_cause, "_deaths")
failures <- paste0("non_", int_cause, "_deaths")
obs_fraction_col <- paste0(int_cause, "_fraction")

source("/ihme/cc_resources/libraries/current/r/get_location_metadata.R")

lasso_cause_list <- function(df) {
  # use lasso to remove causes from the model, only need right side of the formula
  x <- model.matrix(~ cause_id, data = df)
  y <- data.matrix(df[, .SD, .SDcols = c(successes, failures)])
  # https://eight2late.wordpress.com/2017/07/11/a-gentle-introduction-to-logistic-regression-and-lasso-regularisation-using-r/
  # https://cran.r-project.org/web/packages/glmnet/glmnet.pdf
  print(paste0(Sys.time(), " Running lasso"))
  lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial", type.measure = "mse", nfolds = 10)
  print(lasso)

  # save plot output
  print("Saving lambda plot")
  png(paste0(diag_dir, "/lambda_plot.png"))
  plot(lasso)
  dev.off()

  # save fit
  print("Saving plot of the glmnet fit")
  png(paste0(diag_dir, "/fit_plot.png"))
  plot(lasso$glmnet.fit, label = TRUE)
  dev.off()

  # get the coefficients
  betas_lambda_min <- coef(lasso, s = lasso$lambda.min)
  betas_lambda_1se <- coef(lasso, s = lasso$lambda.1se)

  # pick which set of betas to use
  betas <- betas_lambda_1se

  # save betas from all lambdas as diagnostics
  write.csv(as.matrix(betas_lambda_min), file = paste0(data_dir, "/betas_min_lambda.csv"))
  write.csv(as.matrix(betas_lambda_1se), file = paste0(data_dir, "/betas_1se_lambda.csv"))

  # keep the non-zero coefficients
  print(paste0(Sys.time(), " Done! Dropping zero-coefficient causes from the data"))
  # DOCUMENTING:
  # if you need to re run: lasso takes FOREVER, so kept lasso list from previous run
  # betas <- fread(paste0("/ihme/cod/prep/mcod/process_data/", int_cause, "/2019_04_15/betas_1se_lambda.csv"))
  # keep_causes <- betas$V1[which(betas$V2 != 0)]
  # comment out this line below if you need to re run
  keep_causes <- rownames(betas)[which(!betas == 0)]
  keep_causes <- keep_causes[keep_causes %like% "cause_id"]
  keep_causes <- unlist(strsplit(keep_causes, "cause_id"))
  keep_causes <- keep_causes[keep_causes != ""]

  print(paste0("Keeping ", length(keep_causes), " causes"))
  if (length(keep_causes) == 0) {
    print("AH! No causes left after lasso")
  }
  else {
    keep_causes <- noquote(keep_causes)
  }
  return(keep_causes)
}

get_covariates <- function() {
  covariates_df <- fread(paste0(Sys.getenv("HOME"), "/cod-data/mcod_prep/maps/covariates.csv"))
  int_cause_str <- int_cause
  covariates <- covariates_df[covariates_df$int_cause == int_cause_str, covariates]
  covariates <- unlist(strsplit(covariates, ", "))
  return(covariates)
}

get_formula <- function(covariates) {
  base_covariates <- c("sex_id", "age_group_id", "(1|cause_id)")
  exp_vars <- c(covariates, base_covariates)
  resp_vars <- c(successes, failures)
  formula <- reformulate(exp_vars, parse(text = sprintf("cbind(%s)", toString(resp_vars)))[[1]])
  return(formula)
}

run_model <- function(data, covariates) {
  print(paste0(Sys.time(), " Running mixed effect binomial model for ", int_cause))
  formula <- get_formula(covariates)
  model <- glmer(
    formula = formula, data = data, family = binomial(link = "logit"), 
    verbose = 1, nAGQ = 1, control = glmerControl(optimizer = "Nelder_Mead", 
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

save_diagnostics <- function(model, df, covariates) {
  print(paste0(Sys.time(), " Saving diagnostics"))
  # predict values for training data and save as a diagnostic compared to data values
  preds <- predict(model)
  df[, preds_logit := (preds)]
  df[, preds := inv.logit(preds)]
  write.csv(df, paste0(diag_dir, "/", int_cause, "_model_diagnostics.csv"), row.names = F)

  # save coefficients
  adjustments <- list()
  beta <- fixef(model)
  beta_se <- se.fixef(model)
  adjustments[[length(adjustments) + 1]] <- data.table(beta = beta, beta_se = beta_se, cv = names(fixef(model)), rmse = get_rmse(df))
  adjustments <- rbindlist(adjustments)
  adjustments[cv == "(Intercept)", beta := inv.logit(beta)]
  adjustments[cv == "(Intercept)", beta_se := inv.logit(beta_se)]
  adjustments[cv != "(Intercept)", exp_beta := exp(beta)]
  adjustments[cv != "(Intercept)", lower := exp(beta - beta_se * 1.96)]
  adjustments[cv != "(Intercept)", upper := exp(beta + beta_se * 1.96)]
  write.csv(adjustments, paste0(diag_dir, "/", int_cause, "_model_coefficients.csv"), row.names = F)

  # get country column
  loc_df <- get_location_metadata(location_set_version_id=420)[, c("location_id", "ihme_loc_id")]
  df <- merge(loc_df, df, by=c("location_id"))
  df$iso3 <- substr(df$ihme_loc_id, 1, 3)

  # plot residuals by cause
  xlab <- paste0("Data (logit), ", obs_fraction_col)
  ylab <- paste0("Prediction (logit), ", obs_fraction_col)
  for (cause in unique(df$cause_id)) {
    # exclude 0s + 1s
    plot_df <- subset(df, (cause_id == cause) & (get(obs_fraction_col) < 1) & (get(obs_fraction_col) > 0))
    plot_df[, obs_fraction_col] <- plot_df[, logit(get(obs_fraction_col))]
    ggplot(plot_df) + 
      geom_point(aes_string(x=obs_fraction_col, y="preds_logit",
                            color='iso3'), size=3, alpha=0.4) +
      labs(color="Country") +
      theme_bw() + ylab(ylab) + xlab(xlab) +
      geom_abline(intercept=0, slope=1, color='grey') +
      theme(legend.position="bottom") + ggtitle("Predictions vs. data intermediate cause fractions")
    ggsave(paste0(diag_dir, "/residuals_scatter_", cause, ".pdf"), width = 11, height = 8.5)
  }

  df$residuals <- resid(model)
  for (covariate in covariates) {
    # histogram
    ggplot(df, aes_string(x=covariate, color="iso3")) + geom_density()
    ggsave(paste0(diag_dir, "/hist_", covariate, ".pdf"), width = 11, height = 8.5)
    # residuals
    ggplot(df) + geom_point(aes_string(x=covariate, y="residuals", color="iso3"), size=3, alpha=0.4) +
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
data$cause_id <- factor(data$cause_id)

if (!(int_cause %in% c("y34", "x59"))) {
  # read in the cause list that makes up most deaths
  top_pct = "80"
  if (int_cause == "hepatic_failure") {
    top_pct = "90"
  }
  top_df <- fread(paste0(data_dir, sprintf("/top_%s_causes.csv", top_pct)))
  top_causes <- top_df$cause_id
  # drop the data that has these causes
  lasso_df <- data[!(data$cause_id %in% top_causes)]
  # run lasso on the causes that make up the bottom 20% of deaths
  lasso_causes <- lasso_cause_list(lasso_df)
  # read in hard list-- if lasso does not pick these, then add them
  overrides_df <- fread(paste0(Sys.getenv("HOME"), "/cod-data/mcod_prep/maps/lasso_overrides.csv"))
  int_cause_str <- int_cause
  overrides <- overrides_df[overrides_df$int_cause == int_cause_str, cause_id]
  keep_causes <- c(lasso_causes, overrides, top_causes)
  # save output, easy way to grab the causes actually used
  write.table(as.data.frame(keep_causes), file = paste0(data_dir, "/cause_list.csv"), quote = FALSE, sep = ",", row.names = FALSE)
  # only keep the causes that we need before running the model
  data <- data[data$cause_id %in% keep_causes]
} 

model <- run_model(data, covariates)

save_diagnostics(model, data, covariates)

