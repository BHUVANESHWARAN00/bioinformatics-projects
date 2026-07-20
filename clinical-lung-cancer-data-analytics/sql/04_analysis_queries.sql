USE clinical_cancer_analytics;

SELECT COUNT(*) AS total_patients
FROM clinical_data;


SELECT ROUND(AVG(diagnosis_age), 2) AS average_age
FROM clinical_data;


SELECT
    sex,
    COUNT(*) AS total_patients
FROM clinical_data
GROUP BY sex;


SELECT
    race_category,
    COUNT(*) AS total_patients
FROM clinical_data
GROUP BY race_category;


SELECT
    ajcc_stage,
    COUNT(*) AS total_patients
FROM clinical_data
GROUP BY ajcc_stage
ORDER BY total_patients DESC;


SELECT
    vital_status,
    COUNT(*) AS total_patients
FROM clinical_data
GROUP BY vital_status;


SELECT
    overall_survival_status,
    COUNT(*) AS total_patients
FROM clinical_data
GROUP BY overall_survival_status;


SELECT
    ROUND(AVG(years_smoked),2) AS average_years_smoked
FROM clinical_data;

-
SELECT
    ROUND(AVG(mutation_count),2) AS average_mutation_count,
    MAX(mutation_count) AS maximum_mutation_count,
    MIN(mutation_count) AS minimum_mutation_count
FROM clinical_data;


SELECT
    patient_id,
    mutation_count
FROM clinical_data
ORDER BY mutation_count DESC
LIMIT 10;

-- ============================================
-- Query 11: Average Survival by Cancer Stage
-- ============================================

SELECT
    ajcc_stage,
    ROUND(AVG(overall_survival_months),2) AS average_survival_months
FROM clinical_data
GROUP BY ajcc_stage
ORDER BY average_survival_months DESC;


SELECT
    vital_status,
    ROUND(AVG(mutation_count),2) AS average_mutation_count
FROM clinical_data
GROUP BY vital_status;


SELECT
    sex,
    ROUND(AVG(years_smoked),2) AS average_years_smoked
FROM clinical_data
GROUP BY sex;


SELECT
    patient_id,
    diagnosis_age,
    ajcc_stage
FROM clinical_data
WHERE diagnosis_age > 65;


SELECT
    patient_id,
    diagnosis_age,
    sex,
    overall_survival_months
FROM clinical_data
WHERE ajcc_stage='Stage IV';


SELECT
    patient_id,
    overall_survival_months
FROM clinical_data
ORDER BY overall_survival_months DESC
LIMIT 10;


SELECT
    patient_id,
    fraction_genome_altered
FROM clinical_data
ORDER BY fraction_genome_altered DESC
LIMIT 10;