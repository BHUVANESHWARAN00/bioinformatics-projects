library(survival)
library(survminer)
library(ggplot2)
library(dplyr)
library(readr)


clinical_data = read_csv("../data/processed/luad_clinical_cleaned.csv")

head(clinical_data)

dim(clinical_data)

summary(clinical_data)

str(clinical_data)

colSums(is.na(clinical_data))


# Rename Columns
names(clinical_data) = c(
  "patient_id",
  "diagnosis_age",
  "sex",
  "race_category",
  "ethnicity_category",
  "years_smoked",
  "pack_years",
  "ajcc_pathologic_stage",
  "ajcc_pathologic_t_stage",
  "ajcc_pathologic_n_stage",
  "ajcc_pathologic_m_stage",
  "primary_diagnosis",
  "overall_survival_months",
  "overall_survival_status",
  "disease_free_months",
  "disease_free_status",
  "mutation_count",
  "fraction_genome_altered",
  "vital_status",
  "year_of_diagnosis",
  "age_group"
)

survival_data = clinical_data %>%
  select(
    patient_id,
    overall_survival_months,
    overall_survival_status,
    ajcc_pathologic_stage,
    sex,
    diagnosis_age
  )

head(survival_data)


# Create Event Variable
# 1 = Deceased
# 0 = Living

survival_data = survival_data %>%
  mutate(event = ifelse(overall_survival_status == "Deceased",1,0))

table(survival_data$event)

# Remove Missing Survival Time
survival_data = survival_data %>%
  filter(!is.na(overall_survival_months))

dim(survival_data)

# Create Survival Object

surv_object = Surv(
  time = survival_data$overall_survival_months,
  event = survival_data$event)

surv_object

# Kaplan-Meier Model

km_fit = survfit(
  surv_object ~ 1,
  data = survival_data
)

summary(km_fit)

# Kaplan-Meier Survival Curve


km_plot = ggsurvplot(
  km_fit,
  data = survival_data,
  conf.int = TRUE,
  risk.table = TRUE,
  xlab = "Overall Survival (Months)",
  ylab = "Survival Probability",
  title = "Kaplan-Meier Survival Curve - TCGA LUAD",
  ggtheme = theme_bw()
)

print(km_plot)

ggsave(
  filename = "../images/survival/kaplan_meier_curve.png",
  plot = arrange_ggsurvplots(list(km_plot), print = FALSE),
  width = 8,
  height = 8,
  dpi = 300
)



# Survival Analysis by AJCC Stage
stage_data = survival_data %>%
  filter(!is.na(ajcc_pathologic_stage))

table(stage_data$ajcc_pathologic_stage)

stage_fit = survfit(
  Surv(overall_survival_months, event) ~ ajcc_pathologic_stage,
  data = stage_data
)

summary(stage_fit)

stage_plot = ggsurvplot(
  stage_fit,
  data = stage_data,
  pval = TRUE,
  conf.int = FALSE,
  risk.table = TRUE,
  legend.title = "Cancer Stage",
  legend.labs = levels(as.factor(stage_data$ajcc_pathologic_stage)),
  xlab = "Overall Survival (Months)",
  ylab = "Survival Probability",
  title = "Kaplan-Meier Survival by AJCC Pathologic Stage",
  ggtheme = theme_bw()
)

print(stage_plot)

ggsave(
  filename = "../images/survival/kaplan_meier_by_stage.png",
  plot = arrange_ggsurvplots(list(stage_plot), print = FALSE),
  width = 10,
  height = 8,
  dpi = 300
)

# Log-rank Test
log_rank_test = survdiff(
  Surv(overall_survival_months, event) ~ ajcc_pathologic_stage,
  data = stage_data
)

log_rank_test

# Calculate P-value

p_value = 1 - pchisq(
  log_rank_test$chisq,
  length(log_rank_test$n) - 1
)

p_value

# Save Log-rank Test Results
capture.output(
  log_rank_test,
  file = "../reports/log_rank_test_results.txt"
)

writeLines(
  paste("P-value =", p_value),
  "../reports/log_rank_test_pvalue.txt"
)

# Cox Regression Dataset

cox_data = survival_data %>%
  filter(
    !is.na(diagnosis_age),
    !is.na(sex),
    !is.na(ajcc_pathologic_stage)
  )

dim(cox_data)

# Convert Variables to Factors

cox_data$sex = droplevels(as.factor(cox_data$sex))

cox_data$ajcc_pathologic_stage =
  droplevels(as.factor(cox_data$ajcc_pathologic_stage))

str(cox_data)


# Cox Proportional Hazards Model
cox_model = coxph(
  Surv(overall_survival_months, event) ~
    diagnosis_age +
    sex +
    ajcc_pathologic_stage,
  data = cox_data
)

# View Results

summary(cox_model)

# Save Results

capture.output(
  summary(cox_model),
  file = "../reports/cox_regression_results.txt"
)

# Forest Plot

ggforest(
  model = cox_model,
  data = cox_data,
  main = "Hazard Ratio (Cox Regression)",
  fontsize = 0.8
)
ggsave(
  filename = "../images/survival/cox_forest_plot.png",
  width = 10,
  height = 7,
  dpi = 300
)
