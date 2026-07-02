# Breast Cancer Differential Gene Expression Analysis
# Step 3: Differential Expression Analysis

rm(list = ls())

library(limma)

expr = readRDS("data/Expression_Matrix.rds")

group = readRDS("data/Sample_Group.rds")

design = model.matrix(~0 + group)

colnames(design) = levels(group)

design

fit = lmFit(expr, design)

contrast.matrix = makeContrasts(
  Tumor - Normal,
  levels = design
)

contrast.matrix

fit2 = contrasts.fit(fit, contrast.matrix)

fit2 = eBayes(fit2)

result = topTable(
  fit2,
  adjust.method = "BH",
  number = Inf
)

head(result)

dim(result)

write.csv(
  result,
  "results/Differential_Expression_Results.csv",
  row.names = TRUE
)

saveRDS(
  result,
  "results/Differential_Expression_Results.rds"
)

cat("\nStep 3 Completed Successfully\n")