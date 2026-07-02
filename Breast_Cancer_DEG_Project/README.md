# Breast Cancer Differential Gene Expression Analysis

## About this project

This project looks at gene expression differences between normal breast tissue and breast tumor tissue, using a public microarray dataset from NCBI GEO (GSE70947). The goal was to identify genes that are meaningfully up- or down-regulated in tumor samples, and to get a sense of which biological processes those genes are involved in.

I built this as an end-to-end analysis in R, going from raw data download all the way to gene ontology enrichment, and used it as a way to practice a full bioinformatics workflow rather than just running one isolated script.

## Dataset

- **Accession:** GSE70947
- **Platform:** GPL13607
- **Source:** NCBI Gene Expression Omnibus
- **Samples:** 148 normal, 148 tumor (296 total)

## Tools and packages

Built in R/RStudio using packages from Bioconductor and CRAN:

- `GEOquery`, `Biobase` – downloading and handling the GEO dataset
- `limma` – differential expression analysis
- `EnhancedVolcano` – volcano plot
- `pheatmap` – heatmap of top genes
- `ggplot2` – PCA visualization
- `clusterProfiler`, `org.Hs.eg.db` – gene ontology enrichment

## Workflow

1. Downloaded and loaded the GEO series matrix
2. Cleaned the data and removed probes with missing values
3. Defined normal vs. tumor sample groups
4. Ran differential expression analysis with `limma` (empirical Bayes moderated t-tests)
5. Filtered for significant genes (adjusted p-value < 0.05, |log2FC| > 1)
6. Visualized results with a volcano plot, heatmap, and PCA plot
7. Mapped probe IDs to gene symbols using the platform annotation
8. Ran GO enrichment analysis on the significant gene list

KEGG pathway analysis was attempted as well, but the KEGG REST API was unreliable during this run, so that step was made optional — the script checks if the server is reachable and skips gracefully if not, rather than failing the whole pipeline.

## Results

After filtering out probes with missing data, 9,356 genes remained for analysis. Of these:

- **646** were significantly differentially expressed between tumor and normal tissue
- **325** were upregulated in tumor samples
- **321** were downregulated in tumor samples

The PCA plot showed a reasonably clear separation between normal and tumor samples along the first principal component, with some overlap — which is expected in real tissue data, since not every tumor sample looks the same and some normal samples share expression features with early-stage tumors.

GO enrichment on the significant genes pointed to processes like:

- Mitotic nuclear division
- Regulation of nuclear division
- Mitotic sister chromatid separation
- Cell–substrate adhesion

These all fit the expected biology of cancer: tumor cells divide more actively than normal cells, and changes in how cells adhere to their surroundings are commonly linked to tumor invasion.

## Figures

- `figures/Volcano_Plot.png` – significant genes highlighted by fold change and significance
- `figures/Heatmap.pdf` – expression patterns of top differentially expressed genes across samples
- `figures/PCA_Plot.pdf` – sample separation based on overall expression profile
- `figures/GO_Dotplot.pdf`, `figures/GO_Barplot.pdf` – top enriched biological processes

## Project structure

```
Breast_Cancer_DEG_Project/
│
├── data/
│   ├── GSE70947_series_matrix.txt.gz
│   ├── GSE70947_ExpressionSet.rds
│   ├── Expression_Matrix.rds
│   └── Sample_Group.rds
│
├── scripts/
│   ├── 01_Load_Data.R
│   ├── 02_Preprocessing.R
│   ├── 03_Differential_Expression.R
│   ├── 04_Significant_Genes.R
│   ├── 05_Volcano_Plot.R
│   ├── 06_Heatmap.R
│   ├── 07_PCA_Plot.R
│   ├── 08_Gene_Annotation.R
│   ├── 09_GO_Analysis.R
│   ├── 10_KEGG_Analysis.R
│   └── 11_Project_Summary.R
│
├── results/
│   ├── Differential_Expression_Results.csv
│   ├── Significant_Genes.csv
│   ├── Upregulated_Genes.csv
│   ├── Downregulated_Genes.csv
│   ├── Annotated_Differential_Expression_Results.csv
│   ├── Annotated_Significant_Genes.csv
│   ├── GO_Enrichment_Results.csv
│   └── Project_Summary.csv
│
├── figures/
│   ├── Volcano_Plot.png
│   ├── Heatmap.pdf
│   ├── PCA_Plot.pdf
│   ├── GO_Dotplot.pdf
│   └── GO_Barplot.pdf
│
└── README.md
```

## Skills Demonstrated

- R Programming
- Bioconductor
- limma
- Differential Gene Expression Analysis
- Microarray Data Analysis
- Data Visualization
- Gene Annotation
- Gene Ontology Enrichment Analysis

## Notes and limitations

- This is microarray data, not RNA-seq, so the analysis relies on `limma` rather than count-based tools like `DESeq2`.
- About 22% of probe IDs couldn't be mapped to a gene symbol during annotation, which is fairly normal for older microarray platforms.
- KEGG pathway results aren't included here since the KEGG server wasn't reachable at the time of analysis; the GO enrichment results cover similar ground from a different angle.

## Author

**Bhuvaneshwaran G**
B.Tech Biotechnology
