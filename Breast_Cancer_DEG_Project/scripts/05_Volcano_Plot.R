# Breast Cancer Differential Gene Expression Analysis
# Step 5: Volcano Plot

rm(list = ls())

library(EnhancedVolcano)

result = readRDS("results/Differential_Expression_Results.rds")

EnhancedVolcano(
  result,
  lab = rownames(result),
  x = "logFC",
  y = "adj.P.Val",
  title = "Differential Gene Expression",
  subtitle = "Breast Cancer: Tumor vs Normal",
  pCutoff = 0.05,
  FCcutoff = 1,
  pointSize = 2.5,
  labSize = 3.0
)

png(
  filename = "figures/Volcano_Plot.png",
  width = 1800,
  height = 1500,
  res = 300
)

EnhancedVolcano(
  result,
  lab = rownames(result),
  x = "logFC",
  y = "adj.P.Val",
  title = "Differential Gene Expression",
  subtitle = "Breast Cancer: Tumor vs Normal",
  pCutoff = 0.05,
  FCcutoff = 1,
  pointSize = 2.5,
  labSize = 3.0
)

dev.off()

cat("\nStep 5 Completed Successfully\n")