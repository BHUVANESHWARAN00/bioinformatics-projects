# Breast Cancer Differential Gene Expression Analysis
# Step 4: Identify Significant Genes

rm(list = ls())

result = readRDS("results/Differential_Expression_Results.rds")

sigGenes = result[
  abs(result$logFC) > 1 &
    result$adj.P.Val < 0.05,
]

dim(sigGenes)

upGenes = sigGenes[
  sigGenes$logFC > 1,
]

downGenes = sigGenes[
  sigGenes$logFC < -1,
]

cat("Total Significant Genes :", nrow(sigGenes), "\n")
cat("Upregulated Genes :", nrow(upGenes), "\n")
cat("Downregulated Genes :", nrow(downGenes), "\n")

write.csv(
  sigGenes,
  "results/Significant_Genes.csv",
  row.names = TRUE
)

write.csv(
  upGenes,
  "results/Upregulated_Genes.csv",
  row.names = TRUE
)

write.csv(
  downGenes,
  "results/Downregulated_Genes.csv",
  row.names = TRUE
)

saveRDS(
  sigGenes,
  "results/Significant_Genes.rds"
)

cat("\nStep 4 Completed Successfully\n")