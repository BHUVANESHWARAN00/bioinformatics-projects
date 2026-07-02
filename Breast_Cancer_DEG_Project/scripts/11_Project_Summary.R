# Breast Cancer Differential Gene Expression Analysis
# Step 11: Project Summary

rm(list = ls())

result = readRDS("results/Differential_Expression_Results.rds")

sigGenes = readRDS("results/Significant_Genes.rds")

annotated_sigGenes = readRDS("results/Annotated_Significant_Genes.rds")

group = readRDS("data/Sample_Group.rds")

summary_table = data.frame(
  
  Metric = c(
    "Total Samples",
    "Normal Samples",
    "Tumor Samples",
    "Genes Before Filtering",
    "Genes After Filtering",
    "Significant Genes",
    "Upregulated Genes",
    "Downregulated Genes"
  ),
  
  Value = c(
    length(group),
    sum(group == "Normal"),
    sum(group == "Tumor"),
    62976,
    nrow(result),
    nrow(sigGenes),
    sum(sigGenes$logFC > 1),
    sum(sigGenes$logFC < -1)
  )
  
)

print(summary_table)

write.csv(
  summary_table,
  "results/Project_Summary.csv",
  row.names = FALSE
)

 cat("\nProject Summary Saved Successfully\n")