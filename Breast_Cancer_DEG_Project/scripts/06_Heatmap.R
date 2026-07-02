# Breast Cancer Differential Gene Expression Analysis
# Step 6: Heatmap

rm(list = ls())

library(pheatmap)

expr = readRDS("data/Expression_Matrix.rds")

group = readRDS("data/Sample_Group.rds")

result = readRDS("results/Differential_Expression_Results.rds")

# Select Top 50 Significant Genes
top50 = rownames(result)[1:50]

# Extract Expression Values
heatmap_data = expr[top50, ]

# Standardize Expression Values (Z-score)
heatmap_data = t(scale(t(heatmap_data)))

# Sample Annotation
annotation_col = data.frame(Group = group)

rownames(annotation_col) = colnames(heatmap_data)

# Display Heatmap in RStudio
pheatmap(
  heatmap_data,
  annotation_col = annotation_col,
  show_rownames = TRUE,
  show_colnames = FALSE,
  fontsize_row = 8,
  main = "Top 50 Differentially Expressed Genes"
)

# Save Heatmap as PDF
pheatmap(
  heatmap_data,
  annotation_col = annotation_col,
  show_rownames = TRUE,
  show_colnames = FALSE,
  fontsize_row = 8,
  main = "Top 50 Differentially Expressed Genes",
  filename = "figures/Heatmap.pdf",
  width = 12,
  height = 14
)

cat("\nStep 6 Completed Successfully\n")