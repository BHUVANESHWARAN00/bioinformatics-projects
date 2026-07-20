# Clinical Lung Cancer Data Analytics

An end-to-end clinical data analytics project for Lung Adenocarcinoma (LUAD) using MySQL, Python, R, and Power BI. The project demonstrates data preprocessing, exploratory data analysis, survival analysis, SQL-based querying, and interactive dashboard development to derive meaningful clinical insights.

---

## Project Overview

This project analyzes TCGA Lung Adenocarcinoma (LUAD) clinical data to understand patient demographics, tumor characteristics, smoking behavior, genomic alterations, and survival outcomes.

The workflow integrates multiple technologies to perform data management, statistical analysis, and interactive visualization.

---

## Technologies Used

- MySQL – Database creation and SQL queries
- Python – Data cleaning, preprocessing, and exploratory data analysis
- R – Survival analysis (Kaplan–Meier, Cox Regression, Statistical Tests)
- Power BI – Interactive dashboards and data visualization
- DAX – Custom measures and KPIs

---

## Project Workflow

```
Clinical Dataset
        │
        ▼
      MySQL
(Database Management)
        │
        ▼
      Python
(Data Cleaning & EDA)
        │
        ▼
         R
(Survival Analysis)
        │
        ▼
     Power BI
(Interactive Dashboard)
        │
        ▼
 Clinical Insights
```

---

## Project Structure

```
clinical-lung-cancer-data-analytics/
│
├── dashboard/
│   └── clinical-data-analysis.pbix
│
├── data/
│   ├── raw/
│   └── processed/
│
├── python/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_exploratory_data_analysis.ipynb
│
├── r_analysis/
│   ├── 01_survival_analysis.R
│   └── 02_statistical_tests.R
│
├── sql/
│   ├── 01_create_database.sql
│   ├── 02_create_tables.sql
│   ├── 03_insert_data.sql
│   ├── 04_analysis_queries.sql
│   └── 05_views.sql
│
├── reports/
├── images/
└── README.md
```

---

## Dashboard Pages

### Page 1 – Clinical Overview
- Patient demographics
- Age distribution
- Cancer stage distribution
- Survival status

### Page 2 – Smoking & Genomic Analysis
- Smoking exposure
- Mutation burden
- Genome alteration
- Smoking vs mutation analysis

### Page 3 – Patient Survival Analysis
- Kaplan–Meier Survival Curve
- Survival distribution
- Survival status
- Clinical interpretation

### Page 4 – Tumor Characteristics
- Clinical staging
- AJCC analysis
- TNM stage distribution
- Decomposition Tree

### Page 5 – Executive Summary
- Project workflow
- Key findings
- Project conclusion
- Executive KPIs

---

## Key Analyses

- Clinical data preprocessing
- Exploratory Data Analysis (EDA)
- SQL-based cohort analysis
- Smoking exposure analysis
- Mutation burden analysis
- Kaplan–Meier survival analysis
- Cox proportional hazards regression
- Statistical hypothesis testing
- Interactive Power BI dashboards

---

## Key Findings

- A total of 569 LUAD patients were analyzed.
- Most patients were diagnosed between 60–80 years of age.
- Stage IA and Stage IB were the most common AJCC stages.
- Higher smoking exposure was associated with increased mutation burden.
- Most patients were classified as M0, indicating no distant metastasis.
- Kaplan–Meier analysis demonstrated a gradual decline in overall survival probability over time.

---

## How to Run

### Python

Run the Jupyter notebooks in order:

1. `python/01_data_understanding.ipynb`
2. `python/02_data_cleaning.ipynb`
3. `python/03_exploratory_data_analysis.ipynb`

### R

```bash
Rscript r_analysis/01_survival_analysis.R
Rscript r_analysis/02_statistical_tests.R
```

### SQL

Execute the SQL scripts in sequence:

1. `sql/01_create_database.sql`
2. `sql/02_create_tables.sql`
3. `sql/03_insert_data.sql`
4. `sql/04_analysis_queries.sql`
5. `sql/05_views.sql`

### Power BI

Open the dashboard file in the `dashboard/` directory using Microsoft Power BI Desktop:

- `dashboard/clinical-data-analysis.pbix`

---

## Dataset

- TCGA Lung Adenocarcinoma (LUAD) Clinical Dataset

---

## Author

Bhuvaneshwaran G

B.Tech Biotechnology

Kumaraguru College of Technology

---