# NGS Variant Calling Pipeline

This project demonstrates a simple Next Generation Sequencing (NGS) variant calling workflow.

## Tools Used
FastQC  
fastp  
BWA  
SAMtools  
BCFtools  

## Workflow

FASTQ → Quality Control → Trimming → Alignment → Variant Calling

## Reference Genome
Escherichia coli (E. coli)

## Output
Variant file generated in VCF format.

## Project Structure

alignment/
data/
qc/
reference/
trimmed/
variants/


## Workflow Diagram

```mermaid
flowchart TD
    A[FASTQ Reads] --> B[FastQC Quality Check]
    B --> C[fastp Read Trimming]
    C --> D[BWA Alignment]
    D --> E[SAMtools Sorting and Indexing]
    E --> F[BCFtools Variant Calling]
    F --> G[VCF Variant Output]
```
