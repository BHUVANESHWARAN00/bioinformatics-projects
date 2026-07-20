USE clinical_cancer_analytics;

CREATE VIEW survival_summary AS
SELECT
    patient_id,
    ajcc_stage,
    overall_survival_months,
    overall_survival_status,
    vital_status
FROM clinical_data;

CREATE VIEW smoking_summary AS
SELECT
    patient_id,
    sex,
    years_smoked,
    pack_years
FROM clinical_data;

CREATE VIEW mutation_summary AS
SELECT
    patient_id,
    mutation_count,
    fraction_genome_altered
FROM clinical_data;