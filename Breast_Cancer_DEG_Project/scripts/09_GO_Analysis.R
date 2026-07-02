# Breast Cancer Differential Gene Expression Analysis
# Step 9: Gene Ontology (GO) Analysis

rm(list = ls())

library(clusterProfiler)
library(org.Hs.eg.db)

annotated_sigGenes = readRDS("results/Annotated_Significant_Genes.rds")

# Convert Gene Symbols to Entrez IDs
gene.df = bitr(
  annotated_sigGenes$GeneName,
  fromType = "SYMBOL",
  toType = "ENTREZID",
  OrgDb = org.Hs.eg.db
)

# Perform GO Enrichment Analysis
go_result = enrichGO(
  gene = gene.df$ENTREZID,
  OrgDb = org.Hs.eg.db,
  ont = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.05,
  readable = TRUE
)

# View Results
head(as.data.frame(go_result))

# Save Results
write.csv(
  as.data.frame(go_result),
  "results/GO_Enrichment_Results.csv",
  row.names = FALSE
)

saveRDS(
  go_result,
  "results/GO_Enrichment_Results.rds"
)

# Dot Plot
pdf(
  "figures/GO_Dotplot.pdf",
  width = 10,
  height = 8
)

dotplot(
  go_result,
  showCategory = 15,
  title = "GO Biological Process Enrichment"
)

dev.off()

# Bar Plot
pdf(
  "figures/GO_Barplot.pdf",
  width = 10,
  height = 8
)

barplot(
  go_result,
  showCategory = 15,
  title = "GO Biological Process Enrichment"
)

dev.off()

cat("\nStep 9 Completed Successfully\n")