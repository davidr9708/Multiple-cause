.libPaths("/snfs1/temp/scj13/r_packages/3.5")
library(data.table)
library(boot)
library(tidyr)
library(arm)
library(merTools)
library(utils)

# read in sys args
args <- commandArgs(trailingOnly = T)
cause_id <- args[1]
description <- as.character(args[2])
int_cause <- as.character(args[3])
year_id <- args[4]
end_product <- as.character(args[5])

# set base directory
dir <- paste0("/ihme/cod/prep/mcod/process_data/", int_cause, "/", end_product, "/", description, "/")

# read in model path
model <- readRDS(paste0(dir, int_cause, "_model.rds"))

# create template using python, save one result and read it in here
template <- fread(paste0(dir, year_id, '/', cause_id, '_template.csv'))
template$age_group_id <- factor(template$age_group_id)
template$sex_id <- factor(template$sex_id)
if ("cause_id" %in% colnames(template)) {
    template$cause_id <- factor(template$cause_id)
    } else if ("level_2" %in% colnames(template)) {
        template$level_2 <- factor(template$level_2)
        template$level_1 <- factor(template$level_1)
    }

print(paste0(Sys.time(), " Getting predictions for 1000 simluations"))
predictions <- predictInterval(merMod=model, newdata=template, level=0.95, n.sims=1000, stat="mean", 
                               type="probability", returnSims=T, .parallel=TRUE, include.resid.var=F)


sims <- attr(predictions, "sim.results")
sims <- inv.logit(sims)
predictions <- as.data.table(cbind(predictions,sims))
draws <- paste0("draw_", 0:999)
setnames(predictions, c(4:1003), draws)

# add back to template before saving
predictions <- cbind(template, predictions)

# save draws
print(paste0(Sys.time(), " Saving output"))
write.csv(predictions, paste0(dir, year_id, "/", cause_id, ".csv"), row.names=F)
