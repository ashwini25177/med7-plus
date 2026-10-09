# 🏥 Med7 & Med7-Plus: Comprehensive Clinical Information Extraction Study Report
**From Electronic Health Records (EHR) to Clinical Transformers, Medication Slot Linking, and HL7 FHIR Interoperability**

---

## 📋 Executive Summary
This report provides a comprehensive, educational, and technical breakdown of **Clinical Named Entity Recognition (NER)** and **Medication Information Extraction**. It starts from first principles (why medical language processing is difficult), conducts a line-by-line review of the foundational paper **Med7** (*Kormilitzin et al., Artificial Intelligence in Medicine, 2021*), evaluates the limitations of the original model, and details the architectural innovations introduced in **Med7-Plus**.

---

## 📑 Table of Contents
1. [The Foundational Problem: The Electronic Health Record (EHR) Challenge](#1-the-foundational-problem-the-electronic-health-record-ehr-challenge)
2. [Deep Dive: The Original Med7 Research Paper (2021)](#2-deep-dive-the-original-med7-research-paper-2021)
   - 2.1 Research Background & Problem Formulation
   - 2.2 Datasets Used (MIMIC-III, n2c2 2018, CRIS)
   - 2.3 The Two-Stage Training Paradigm
   - 2.4 The 7 Target Categories
   - 2.5 Clinical Evaluation Methodology (Strict vs. Lenient F1)
   - 2.6 The Key Finding: Cross-Country Domain Transferability
3. [Limitations of Original Med7: The Motivation for Med7-Plus](#3-limitations-of-original-med7-the-motivation-for-med7-plus)
4. [Med7-Plus Architecture & Key Improvements](#4-med7-plus-architecture--key-improvements)
   - 4.1 Medication Slot Linking (Relation Extraction)
   - 4.2 Context & Negation Analysis (NegEx + n2c2 2022 CMED)
   - 4.3 Drug Concept Normalization (RxNorm & ATC Classification)
   - 4.4 Health IT Standards (HL7 FHIR R4 Interoperability)
   - 4.5 Dual-Engine Architecture & Synthetic Corpus
5. [Step-by-Step Practical Guide: Setup, Execution, and Cloud Deployment](#5-step-by-step-practical-guide-setup-execution-and-cloud-deployment)
6. [Mathematical Formulation: Clinical NER Metrics](#6-mathematical-formulation-clinical-ner-metrics)
7. [End-to-End Case Study: Clinical Note to FHIR JSON](#7-end-to-end-case-study-clinical-note-to-fhir-json)
8. [Bibliography & Primary Sources](#8-bibliography--primary-sources)

---

## 1. The Foundational Problem: The Electronic Health Record (EHR) Challenge

### 1.1 The Dark Data of Healthcare
Modern hospital systems generate massive amounts of Electronic Health Record (EHR) data. However, **over 80% of actionable clinical information exists exclusively as unstructured free-text narratives**:
- Physician progress notes
- Nursing shift reports
- Discharge summaries
- Operative logs and radiology reports

While hospital billing systems store structured tables (ICD-10 diagnostic codes and billing records), **critical nuances are lost in tabular data**:
- Did the patient actually take the prescribed medication, or was it withheld?
- What was the specific dosage schedule (e.g., *tapering down over 14 days*)?
- Did the patient discontinue the medication due to an adverse allergic reaction?

### 1.2 Why Standard NLP Tools Fail on Clinical Text
Standard general-domain NLP tools (trained on Wikipedia, news articles, or web crawl datasets) fail on clinical narratives due to unique linguistic properties:
1. **Pervasive Medical Shorthand & Latin Abbreviations**:
   - `PO` (*per os* — by mouth / oral)
   - `BID` (*bis in die* — twice daily)
   - `QHS` (*quaque hora somni* — every night at bedtime)
   - `PRN` (*pro re nata* — as needed)
2. **Inverted and Compressed Dosage Syntax**:
   - *General English*: "The doctor gave John two pills every morning."
   - *Clinical EHR*: "Metformin 500mg PO BID with meals x 14d."
3. **Compound Numerical Spans & Complex Units**:
   - Continuous infusions: `18 units/kg/hr`, `0.5 mcg/min`, `5 mg/ml`.
4. **Heavy Punctuation Overload**:
   - In clinical text, periods are used for abbreviations (`p.o.`, `q.12h.`) rather than sentence boundaries, breaking standard tokenizers.
5. **Critical Negation Dynamics**:
   - *"Patient denies chest pain. Denies taking Aspirin at home."* A general entity recognizer finds `Aspirin`, but misclassifying it as an active prescription could endanger a patient.

---

## 2. Deep Dive: The Original Med7 Research Paper (2021)

### 2.1 Bibliographic Details
- **Title**: *Med7: a transferable clinical natural language processing model for electronic health records*
- **Authors**: Andrey Kormilitzin, Nemanja Vaci, Qiang Liu, Alejo Nevado-Holgado
- **Affiliation**: University of Oxford, Department of Psychiatry; National Institute for Health Research (NIHR) Oxford Health Biomedical Research Centre.
- **Journal**: *Artificial Intelligence in Medicine*, Vol. 118, Article 102086 (August 2021).
- **Preprint**: [arXiv:2003.01271](https://arxiv.org/abs/2003.01271)

### 2.2 Datasets Used in the Study

```
┌────────────────────────────────────────────────────────┐
│                   MIMIC-III Corpus                     │
│    (2 Million Free-Text Patient Records from US ICU)    │
└──────────────────────────┬─────────────────────────────┘
                           │
             Self-Supervised Pre-Training
             (Next-Word Language Modeling)
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                   Fine-Tuning Data                     │
│  ├─ n2c2 2018 Gold Annotations (Track 2 Benchmark)     │
│  ├─ 606 MIMIC-III Documents (Manually Annotated)      │
│  └─ 303 MIMIC-III Documents (Silver Rule-Annotated)    │
└──────────────────────────┬─────────────────────────────┘
                           │
                    Supervised NER
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                    Med7 Model                          │
│          (7 Medication Entities, F1=0.957)             │
└──────────────────────────┬─────────────────────────────┘
                           │
             Domain Transferability Testing
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                      CRIS                              │
│       (UK Secondary Care Mental Health Records)        │
└────────────────────────────────────────────────────────┘
```

#### A. MIMIC-III Database (Pre-training)
- **Source**: Medical Information Mart for Intensive Care III (Beth Israel Deaconess Medical Center, Boston, MA).
- **Scale**: **2,083,180 unstructured clinical notes** from over 40,000 intensive care unit patients.
- **Task**: Self-supervised language representation learning (predicting the next word / causal language modeling) to teach the model clinical grammar, pharmaceutical vocabulary, and clinical sentence structures without requiring manual annotations.

#### B. Supervised Fine-Tuning Corpus (Gold + Silver)
1. **2018 n2c2 Shared Task (Track 2)**: Gold-standard clinical benchmark for adverse drug event and medication extraction released by Harvard Medical School and the National NLP Clinical Challenges.
2. **Prodigy-Annotated MIMIC-III Subset**: **606 clinical notes** manually annotated by clinical domain experts using the Prodigy annotation tool following the official n2c2 2018 guidelines.
3. **Silver Data Subset**: **303 clinical documents** auto-annotated using rule-based pattern matchers in spaCy. The authors demonstrated that augmenting gold data with silver data improved model robustness at low cost.

#### C. CRIS Database (Transferability Target)
- **Source**: Clinical Record Interactive Search (South London and Maudsley NHS Foundation Trust, UK).
- **Domain**: De-identified secondary care psychiatric and mental health records.

### 2.3 The 7 Target Categories (The "Med7" Entities)

| # | Entity Label | Clinical Definition | Concrete Clinical Examples |
|---|:---|:---|:---|
| 1 | **`DRUG`** | Active chemical entity, generic name, or commercial brand name. | *Aspirin*, *Metformin*, *Lisinopril*, *Lipitor* |
| 2 | **`STRENGTH`** | The potency or concentration of active drug substance. | *500 mg*, *40 mg*, *18 units/kg/hr*, *5 %* |
| 3 | **`DOSAGE`** | The number of formulation units administered. | *1 tablet*, *2 puffs*, *1 vial*, *2 capsules* |
| 4 | **`ROUTE`** | The anatomical path or method by which the drug enters the body. | *PO*, *oral*, *IV*, *subcutaneous*, *topical* |
| 5 | **`FREQUENCY`** | The temporal dosing interval or schedule. | *once daily*, *BID*, *TID*, *q8h*, *at bedtime* |
| 6 | **`FORM`** | The physical pharmaceutical delivery formulation. | *tablet*, *capsule*, *solution*, *cream*, *inhaler* |
| 7 | **`DURATION`** | The total elapsed time period for the treatment regimen. | *for 14 days*, *x 4 weeks*, *for 3 months* |

### 2.4 Clinical Evaluation Methodology: Strict vs. Lenient F1
In general NLP (e.g., CoNLL-2003 on news articles), matching is evaluated strictly: character start and end offsets must match the gold annotation identically.

In **clinical medicine**, however, exact boundary matches do not always capture clinical utility. For example:
- **Gold Span**: `[0:20]` -> *"Metoprolol succinate"* (`DRUG`)
- **Predicted Span**: `[0:10]` -> *"Metoprolol"* (`DRUG`)

Under strict scoring, this is penalized as a **False Positive and a False Negative (0% accuracy)**. Yet clinically, the model correctly identified the drug concept, its active molecule, and its therapeutic class.

Therefore, Kormilitzin et al. established two standard evaluation metrics:
- **Strict Matching**: Requires identical label AND identical character offsets ($start_{pred} = start_{gold}$ and $end_{pred} = end_{gold}$).
- **Lenient Matching**: Requires identical label AND a character overlap ($\max(start_1, start_2) < \min(end_1, end_2)$).

#### Performance Results in the Paper:
- **Lenient Micro-averaged F1 Score**: **0.957** (Precision: 0.961, Recall: 0.954)
- **Strict Micro-averaged F1 Score**: **0.893** (Precision: 0.897, Recall: 0.889)

### 2.5 The Key Scientific Finding: Cross-Country Domain Transferability
The most important scientific discovery of Kormilitzin et al. was testing whether a model trained on US intensive care data could transfer to UK psychiatric outpatient data:

$$\text{MIMIC-III (US ICU Data)} \longrightarrow \text{CRIS (UK Mental Health Data)}$$

1. **Direct Zero-Shot Transfer**:
   - Applying Med7 directly onto CRIS records without adaptation resulted in a sharp drop in performance:
   $$\text{Lenient F1} = 0.762 \quad (\downarrow 19.5\%)$$
   - *Why?* Differences in healthcare systems: US discharge summaries emphasize acute cardiology/ICU drugs (heparin, vasopressors, furosemide), whereas UK mental health notes emphasize psychiatric drugs, depot injections, and different clinical abbreviations.
2. **Domain-Adapted Fine-Tuning**:
   - The authors fine-tuned Med7 on just a small sample of CRIS annotations:
   $$\text{Lenient F1} = 0.944 \quad (\text{Performance Recovered!})$$
   - **Conclusion**: Clinical NLP models cannot be blindly transferred across different hospital domains or healthcare systems without fine-tuning on the target domain.

---

## 3. Limitations of Original Med7: The Motivation for Med7-Plus

While Med7 was an outstanding scientific achievement, in real-world healthcare software, several critical gaps remained:

### Gap 1: No Relation Extraction / Slot Linking
Med7 produces a "flat bag" of entity spans. If a clinical note reads:
> *"Patient prescribed Metformin 500mg PO BID for diabetes, and Atorvastatin 20mg PO at bedtime."*

Med7 detects:
`Metformin` (DRUG), `500mg` (STRENGTH), `BID` (FREQUENCY), `Atorvastatin` (DRUG), `20mg` (STRENGTH), `at bedtime` (FREQUENCY).

**The Problem**: It does not tell downstream systems which strength or frequency belongs to which drug! If `500mg` is accidentally linked to `Atorvastatin` (which has a maximum daily dose of 80mg), this would trigger a critical safety alert in an EHR.

### Gap 2: No Context & Negation Handling
Consider this note:
> *"Discontinued Lisinopril due to persistent dry cough. Patient denies taking Warfarin at home."*

Med7 extracts `Lisinopril` and `Warfarin` as `DRUG`. If fed into an automated patient chart, both drugs would appear on the active medication list, misrepresenting discontinued or denied medications as active prescriptions.

### Gap 3: No Ontology Normalization
Med7 outputs raw text strings (*"Tylenol"*, *"Acetaminophen"*, *"APAP"*). Hospital analytics require standardization to **RxNorm Concept Unique Identifiers (RxCUI)** and **Anatomical Therapeutic Chemical (ATC)** classification codes.

### Gap 4: No Health IT Standard Compliance
Med7 outputs Python tuples. Modern digital health systems (Epic, Cerner, hospital data lakes) require **HL7 FHIR R4 (Fast Healthcare Interoperability Resources)** JSON structures.

### Gap 5: High Barrier to Entry
Accessing MIMIC-III or MIMIC-IV requires passing CITI ethics training, obtaining PhysioNet credentialing, and signing a Data Use Agreement (DUA). Without synthetic data, students, recruiters, and developers cannot run the model immediately.

---

## 4. Med7-Plus Architecture & Key Improvements

**Med7-Plus** addresses each of these gaps:

```
┌────────────────────────────────────────────────────────────────────────┐
│                       Unstructured Clinical Note                       │
│ "Furosemide 40 mg PO once daily for 14 days. Discontinued Lisinopril." │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        1. Entity Recognition                           │
│  Extracts Spans: DRUG, STRENGTH, ROUTE, FREQUENCY, DURATION, etc.      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐       ┌────────────────────┐     ┌───────────────────┐
│   2. Context    │       │  3. Normalization  │     │  4. Slot Linker   │
│    Analyzer     │       │       Engine       │     │                   │
│ NegEx/n2c2:     │       │ RxNorm CUI: 4603   │     │ Binds 40mg, PO,   │
│ Lisinopril ->   │       │ ATC: C03CA01       │     │ daily, 14 days    │
│ DISCONTINUED    │       │ Furosemide Generic │     │ to Furosemide     │
└────────┬────────┘       └─────────┬──────────┘     └─────────┬─────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    5. Interoperable Output Generation                  │
│  ├─ Interactive Streamlit UI (DisplaCy Color Highlighting)             │
│  ├─ FastAPI REST Endpoints (/docs, /extract, /health)                 │
│  └─ Standardized HL7 FHIR R4 MedicationStatement Collection            │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Medication Slot Linking (Relation Extraction)
Located in [`med7_plus/relation_linker.py`](file:///C:/Users/Ashwini/OneDrive/Desktop/New%20folder/cell-viewer/med7-clinical-extraction/med7_plus/relation_linker.py).
- Groups extracted modifiers (`STRENGTH`, `DOSAGE`, `ROUTE`, `FREQUENCY`, `DURATION`, `FORM`) with their parent `DRUG`.
- Uses a **character-distance metric with sentence-boundary constraints**:
  $$\text{Distance} = |Start_{\text{mod}} - End_{\text{drug}}| + \text{Penalty}_{\text{sentence}}$$
  Crossing a period (`.`) or newline (`\n`) adds an 80-character distance penalty, preventing modifiers from jumping across clinical sentences.

### 4.2 Clinical Context Analyzer (NegEx + n2c2 2022 CMED)
Located in [`med7_plus/context_analyzer.py`](file:///C:/Users/Ashwini/OneDrive/Desktop/New%20folder/cell-viewer/med7-clinical-extraction/med7_plus/context_analyzer.py).
- **Negation Status**: Identifies pre- and post-entity triggers (*"discontinued"*, *"stopped"*, *"denies"*, *"withheld"*, *"allergic to"*) to classify medications as `ACTIVE` vs. `DISCONTINUED_OR_NEGATED`.
- **Certainty**: Categorizes into `CONFIRMED`, `POSSIBLE` (*"suspected"*, *"evaluate for"*), or `CONDITIONAL` (*"PRN"*, *"as needed"*).
- **Temporality**: Categorizes into `CURRENT`, `PAST`, or `FUTURE` (*"discharge prescription"*).

### 4.3 Drug Concept Normalizer
Located in [`med7_plus/normalizer.py`](file:///C:/Users/Ashwini/OneDrive/Desktop/New%20folder/cell-viewer/med7-clinical-extraction/med7_plus/normalizer.py).
- Resolves free-text mentions into standardized identifiers:
  - *Furosemide* $\rightarrow$ RxNorm CUI: `4603`, ATC Code: `C03CA01` (Loop Diuretic)
  - *Metformin* $\rightarrow$ RxNorm CUI: `6809`, ATC Code: `A10BA02` (Biguanide Antidiabetic)
  - *Lisinopril* $\rightarrow$ RxNorm CUI: `29046`, ATC Code: `C09AA03` (ACE Inhibitor)

### 4.4 HL7 FHIR R4 Interoperability Exporter
Located in [`med7_plus/fhir_exporter.py`](file:///C:/Users/Ashwini/OneDrive/Desktop/New%20folder/cell-viewer/med7-clinical-extraction/med7_plus/fhir_exporter.py).
- Transforms structured extractions into official `MedicationStatement` resources:
  - `status`: `"active"` or `"not-taken"`
  - `medicationCodeableConcept`: Official RxNorm coding
  - `dosage`: Hierarchical `doseAndRate`, `route`, and `timing`
  - `note`: Negation and PRN clinical commentary

### 4.5 Dual-Engine Architecture
Located in [`med7_plus/extractor.py`](file:///C:/Users/Ashwini/OneDrive/Desktop/New%20folder/cell-viewer/med7-clinical-extraction/med7_plus/extractor.py).
- **Neural Engine**: Connects to `en_core_med7_lg` or transformer pipelines (`Bio_ClinicalBERT`) if installed.
- **Rule/Pattern Engine**: Features high-precision regular expressions and clinical lexicons. If no heavy model is downloaded, the system runs with zero errors and immediate execution.

---

## 5. Step-by-Step Practical Guide: Setup, Execution, and Cloud Deployment

### 5.1 Cloning the Repository
```bash
git clone https://github.com/ashwini25177/med7-plus.git
cd med7-plus
```

### 5.2 Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5.3 Running the CLI Demo
```bash
python demo.py
```
Outputs colored token spans, drug slot records, active/discontinued status, and generated FHIR JSON payloads.

### 5.4 Running the Test Suite (10/10 Passing Tests)
```bash
python -m unittest discover tests
```

### 5.5 Launching the Streamlit Web Application
```bash
streamlit run web_app/app_streamlit.py
```
Access at `http://localhost:8501`.

### 5.6 Running the Production FastAPI REST Microservice
```bash
uvicorn web_app.api:app --reload
```
Interactive Swagger UI documentation is available at:
👉 **`http://localhost:8000/docs`**

### 5.7 Cloud Deployment (Streamlit Community Cloud)
1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub (`ashwini25177`).
2. Click **Create app**.
3. Select:
   - Repository: `ashwini25177/med7-plus`
   - Branch: `main`
   - Main file path: `web_app/app_streamlit.py`
4. Click **Deploy!**

---

## 6. Mathematical Formulation: Clinical NER Metrics

Let $G$ be the set of gold-standard annotations and $P$ be the set of model predictions.

Each annotation $a \in G \cup P$ is represented as a tuple:
$$a = (start, end, label)$$

### 6.1 Strict Matching Criterion
A prediction $p \in P$ is a **Strict True Positive ($TP_{strict}$)** if and only if there exists $g \in G$ such that:
$$p.label = g.label \quad \land \quad p.start = g.start \quad \land \quad p.end = g.end$$

### 6.2 Lenient Matching Criterion
A prediction $p \in P$ is a **Lenient True Positive ($TP_{lenient}$)** if and only if there exists $g \in G$ such that:
$$p.label = g.label \quad \land \quad \max(p.start, g.start) < \min(p.end, g.end)$$

### 6.3 Precision, Recall, and Micro-F1 Formulation
For either criterion ($k \in \{\text{strict}, \text{lenient}\}$):

$$\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}$$

$$\text{Recall}_k = \frac{TP_k}{TP_k + FN_k}$$

$$F1_k = \frac{2 \times \text{Precision}_k \times \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

---

## 7. End-to-End Case Study: Clinical Note to FHIR JSON

### Input Clinical Narrative:
> *"Patient admitted with acute decompensated heart failure. Discharge medications: Furosemide 40 mg tablet PO once daily for 14 days. Metoprolol succinate 25 mg PO daily. Discontinued Lisinopril due to dry cough."*

### Step 1: Raw Span Extraction
- `[80:90]` **`DRUG`**: *Furosemide*
- `[91:96]` **`STRENGTH`**: *40 mg*
- `[97:103]` **`FORM`**: *tablet*
- `[104:106]` **`ROUTE`**: *PO*
- `[107:117]` **`FREQUENCY`**: *once daily*
- `[118:129]` **`DURATION`**: *for 14 days*
- `[131:151]` **`DRUG`**: *Metoprolol succinate*
- `[152:157]` **`STRENGTH`**: *25 mg*
- `[158:160]` **`ROUTE`**: *PO*
- `[161:166]` **`FREQUENCY`**: *daily*
- `[181:191]` **`DRUG`**: *Lisinopril*

### Step 2: Context & Slot Linking
- **Event #1**: *Furosemide*
  - Status: `ACTIVE`
  - Linked Slots: Strength=`40 mg`, Route=`PO`, Frequency=`once daily`, Duration=`for 14 days`
  - Normalized: RxNorm CUI `4603`, ATC `C03CA01`
- **Event #2**: *Lisinopril*
  - Status: `DISCONTINUED_OR_NEGATED` (Preceding trigger: *"Discontinued"*)
  - Normalized: RxNorm CUI `29046`, ATC `C09AA03`

### Step 3: Generated HL7 FHIR R4 JSON Statement
```json
{
  "resourceType": "MedicationStatement",
  "id": "med7-furosemide-001",
  "status": "active",
  "subject": {
    "reference": "Patient/PATIENT-DEMO-001"
  },
  "effectiveDateTime": "2026-10-09T23:00:00Z",
  "medicationCodeableConcept": {
    "text": "Furosemide",
    "coding": [
      {
        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
        "code": "4603",
        "display": "Furosemide"
      }
    ]
  },
  "dosage": [
    {
      "doseAndRate": [
        {
          "type": { "text": "ordered strength" },
          "doseQuantity": { "value": "40 mg" }
        }
      ],
      "route": { "text": "PO" },
      "timing": {
        "code": { "text": "once daily" },
        "repeat": { "boundsDuration": { "text": "for 14 days" } }
      }
    }
  ]
}
```

---

## 8. Bibliography & Primary Sources

1. **Kormilitzin, A., Vaci, N., Liu, Q., & Nevado-Holgado, A. (2021).** Med7: A transferable clinical natural language processing model for electronic health records. *Artificial Intelligence in Medicine*, 118, 102086. [DOI: 10.1016/j.artmed.2021.102086](https://doi.org/10.1016/j.artmed.2021.102086) | [arXiv:2003.01271](https://arxiv.org/abs/2003.01271).
2. **Johnson, A. E. W., et al. (2016).** MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3, 160035.
3. **Johnson, A. E. W., et al. (2023).** MIMIC-IV-Note: De-identified free-text clinical notes (version 2.2). *PhysioNet*.
4. **Henry, S., Buchan, K., Filannino, M., Stubbs, A., & Uzuner, Ö. (2020).** 2018 n2c2 shared task on adverse drug events and medication extraction in electronic health records. *Journal of the American Medical Informatics Association*, 27(1), 3–12.
5. **Chapman, W. W., et al. (2001).** A simple algorithm for identifying negated findings and diseases in discharge summaries (NegEx). *Journal of Biomedical Informatics*, 34(5), 301–310.
6. **HL7 International (2019).** HL7 Fast Healthcare Interoperability Resources (FHIR) Release 4: MedicationStatement Resource. [hl7.org/fhir/R4/medicationstatement.html](https://hl7.org/fhir/R4/medicationstatement.html).

---
*Report compiled for the Med7-Plus Open Source Initiative.*  
*Repository: [https://github.com/ashwini25177/med7-plus](https://github.com/ashwini25177/med7-plus)*
