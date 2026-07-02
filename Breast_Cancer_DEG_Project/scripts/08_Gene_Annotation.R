# Breast Cancer Differential Gene Expression Analysis
# Step 8: Gene Annotation

rm(list = ls())

library(GEOquery)

result = readRDS("results/Differential_Expression_Results.rds")

sigGenes = readRDS("results/Significant_Genes.rds")

# Download GPL Annotation
gpl = getGEO("GPL13607")

# Convert GPL object to table
gplTable = Table(gpl)

# Select required columns
annotation = gplTable[, c("ID", "GeneName", "Description")]

# Add Probe ID column
result$ID = rownames(result)

sigGenes$ID = rownames(sigGenes)

# Merge annotation with all genes
annotated_result = merge(
  result,
  annotation,
  by = "ID",
  all.x = TRUE
)

# Merge annotation with significant genes
annotated_sigGenes = merge(
  sigGenes,
  annotation,
  by = "ID",
  all.x = TRUE
)

# Remove rows without Gene Name
annotated_result = annotated_result[
  annotated_result$GeneName != "" &
    !is.na(annotated_result$GeneName),
]

annotated_sigGenes = annotated_sigGenes[
  annotated_sigGenes$GeneName != "" &
    !is.na(annotated_sigGenes$GeneName),
]

# View first few annotated genes
head(annotated_result)

head(annotated_sigGenes)

# Save Results
write.csv(
  annotated_result,
  "results/Annotated_Differential_Expression_Results.csv",
  row.names = FALSE
)

write.csv(
  annotated_sigGenes,
  "results/Annotated_Significant_Genes.csv",
  row.names = FALSE
)

saveRDS(
  annotated_sigGenes,
  "results/Annotated_Significant_Genes.rds"
)

cat("\nStep 8 Completed Successfully\n")