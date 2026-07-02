# Breast Cancer Differential Gene Expression Analysis
# Step 7: Principal Component Analysis (PCA)

rm(list = ls())

library(ggplot2)

expr = readRDS("data/Expression_Matrix.rds")

group = readRDS("data/Sample_Group.rds")

# Transpose the expression matrix
expr_t = t(expr)

# Perform PCA
pca = prcomp(expr_t, scale. = TRUE)

# Create data frame
pca_df = data.frame(
  PC1 = pca$x[,1],
  PC2 = pca$x[,2],
  Group = group
)

# Display PCA Plot
ggplot(
  pca_df,
  aes(
    x = PC1,
    y = PC2,
    color = Group
  )
) +
  geom_point(size = 3) +
  labs(
    title = "PCA Plot",
    subtitle = "Breast Cancer: Tumor vs Normal",
    x = "Principal Component 1",
    y = "Principal Component 2"
  ) +
  theme_minimal(base_size = 14)

# Save PCA Plot as PDF
pdf(
  "figures/PCA_Plot.pdf",
  width = 8,
  height = 6
)

ggplot(
  pca_df,
  aes(
    x = PC1,
    y = PC2,
    color = Group
  )
) +
  geom_point(size = 3) +
  labs(
    title = "PCA Plot",
    subtitle = "Breast Cancer: Tumor vs Normal",
    x = "Principal Component 1",
    y = "Principal Component 2"
  ) +
  theme_minimal(base_size = 14)

dev.off()

cat("\nStep 7 Completed Successfully\n")