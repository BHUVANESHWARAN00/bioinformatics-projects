
library(readr)
library(dplyr)
library(ggplot2)

# Load Dataset

clinical_data = read_csv("../data/processed/luad_clinical_cleaned.csv")

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

# Dataset Summary

head(clinical_data)

dim(clinical_data)

summary(clinical_data)

str(clinical_data)

colSums(is.na(clinical_data))

# Descriptive Statistics

summary(clinical_data$diagnosis_age)

summary(clinical_data$mutation_count)

summary(clinical_data$years_smoked)

summary(clinical_data$pack_years)

capture.output(
  summary(clinical_data),
  file = "../reports/descriptive_statistics.txt"
)

# Pearson Correlation (Age vs Mutation Count)

age_mutation_cor = cor.test(
  clinical_data$diagnosis_age,
  clinical_data$mutation_count,
  method = "pearson"
)

age_mutation_cor

capture.output(
  age_mutation_cor,
  file = "../reports/age_mutation_correlation.txt"
)

# Pearson Correlation (Smoking Years vs Mutation Count)

smoking_mutation_cor = cor.test(
  clinical_data$years_smoked,
  clinical_data$mutation_count,
  method = "pearson"
)

smoking_mutation_cor

capture.output(
  smoking_mutation_cor,
  file = "../reports/smoking_mutation_correlation.txt"
)

# Pearson Correlation (Pack Years vs Mutation Count)

packyear_mutation_cor = cor.test(
  clinical_data$pack_years,
  clinical_data$mutation_count,
  method = "pearson"
)

packyear_mutation_cor

capture.output(
  packyear_mutation_cor,
  file = "../reports/packyear_mutation_correlation.txt"
)

# Independent t-test

ttest_result = t.test(
  mutation_count ~ sex,
  data = clinical_data
)

ttest_result

capture.output(
  ttest_result,
  file = "../reports/t_test_results.txt"
)

# One-Way ANOVA

anova_model = aov(
  mutation_count ~ ajcc_pathologic_stage,
  data = clinical_data
)

summary(anova_model)

capture.output(
  summary(anova_model),
  file = "../reports/anova_results.txt"
)

# Chi-Square Test

stage_table = table(
  clinical_data$sex,
  clinical_data$ajcc_pathologic_stage
)

chi_result = chisq.test(stage_table)

chi_result

capture.output(
  chi_result,
  file = "../reports/chi_square_results.txt"
)

