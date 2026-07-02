# Breast Cancer Differential Gene Expression Analysis
# Step 2: Data Exploration and Preprocessing

rm(list = ls())

library(Biobase)

data = readRDS("data/GSE70947_ExpressionSet.rds")

expr = exprs(data)

dim(expr)

group = ifelse(
  pData(data)$characteristics_ch1.1 == "tissue: normal",
  "Normal",
  "Tumor"
)

group = factor(group)

table(group)

sum(is.na(expr))

expr = expr[complete.cases(expr), ]

dim(expr)

saveRDS(expr, "data/Expression_Matrix.rds")

saveRDS(group, "data/Sample_Group.rds")

cat("\nStep 2 Completed Successfully\n")