# 🔒 Clinical Datasets & Data Use Agreements (DUA)

Real-world clinical Electronic Health Record (EHR) data contains sensitive patient records and cannot be publicly distributed in Git repositories under HIPAA regulations.

## 1. MIMIC-IV / MIMIC-III (MIT Lab for Computational Physiology)
- **Website**: [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/)
- **Access Requirements**:
  1. Complete the CITI "Data or Specimens Only Research" ethics training course.
  2. Create an account on PhysioNet and request credentialed access.
  3. Sign the Data Use Agreement (DUA).
- **File Needed**: `discharge.csv.gz` (MIMIC-IV-Note) or `NOTEEVENTS.csv` (MIMIC-III).
- **Placement**: Place downloaded CSVs in this folder (`data/raw/`) and run:
  ```bash
  python scripts/convert_mimic.py --input data/raw/discharge.csv --output data/processed/mimic_notes.json
  ```

## 2. n2c2 / i2b2 Shared Tasks (Harvard Medical School DBMI)
- **Website**: [Harvard DBMI n2c2 Portal](https://n2c2.dbmi.hms.harvard.edu/)
- **Challenges**:
  - **2018 Track 2**: Adverse Drug Events and Medication Extraction
  - **2022 Track 1**: Contextualized Medication Event Extraction (CMED)
- **Access Requirements**: Free academic/research registration and agreement submission.
- **Conversion**:
  ```bash
  python scripts/convert_n2c2.py --input-dir data/raw/n2c2_2022/ --output data/processed/n2c2_dataset.json
  ```

## 3. Immediate Out-of-the-Box Usage (Zero Credentials Needed)
If you do not have PhysioNet credentials, **you do not need to download anything!**
Med7-Plus includes pre-annotated synthetic clinical records in [`data/sample/synthetic_ehr_notes.json`](../sample/synthetic_ehr_notes.json) which you can use immediately for testing, evaluation, and fine-tuning.
