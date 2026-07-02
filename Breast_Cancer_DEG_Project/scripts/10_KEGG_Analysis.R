# Breast Cancer Differential Gene Expression Analysis
# Step 10: KEGG Pathway Analysis

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

# Perform KEGG Analysis
kegg_result = tryCatch(
  
  enrichKEGG(
    gene = gene.df$ENTREZID,
    organism = "hsa",
    pvalueCutoff = 0.05
  ),
  
  error = function(e) {
    
    cat("\n")
    cat("===============================\n")
    cat("KEGG server is currently unavailable.\n")
    cat("Skipping KEGG Analysis.\n")
    cat("===============================\n")
    
    return(NULL)
    
  }
  
)

# If KEGG worked
if (!is.null(kegg_result)) {
  
  head(as.data.frame(kegg_result))
  
  write.csv(
    as.data.frame(kegg_result),
    "results/KEGG_Enrichment_Results.csv",
    row.names = FALSE
  )
  
  saveRDS(
    kegg_result,
    "results/KEGG_Enrichment_Results.rds"
  )
  
  pdf(
    "figures/KEGG_Dotplot.pdf",
    width = 10,
    height = 8
  )
  
  dotplot(
    kegg_result,
    showCategory = 15,
    title = "KEGG Pathway Enrichment"
  )
  
  dev.off()
  
  pdf(
    "figures/KEGG_Barplot.pdf",
    width = 10,
    height = 8
  )
  
  barplot(
    kegg_result,
    showCategory = 15,
    title = "KEGG Pathway Enrichment"
  )
  
  dev.off()
  
  cat("\nKEGG Analysis Completed Successfully\n")
  
}