---
title: Student Dropout Risk Prediction
emoji: 🎓
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.9.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Student Dropout Risk Prediction

A Logistic Regression classification project and Gradio prototype for educational early warning and student support. Uses an existing trained model; the application never fits or retrains it.

**Repository:** [student-dropout-risk-prediction](https://github.com/Ahsan-Qamar-Dev/student-dropout-risk-prediction)

**Live application:** Deployment pending Render account card verification and public testing.

## Project Overview

This internship project turns academic, enrollment, and financial indicators into an estimated dropout probability. Its purpose is to demonstrate an educational early-warning workflow, with human review and support as the intended follow-up.

## Problem Statement

Predict whether a student belongs to the dropout class. Target encoding is **0 = Not Dropout** (Graduate in the training cohort) and **1 = Dropout**. Students still enrolled were excluded from the binary training dataset; “Not Dropout” must not be interpreted as a guarantee of future graduation.

## Dataset

The notebook and supplied preprocessed CSV match [UCI Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success). It contains enrollment, demographic, socioeconomic, and first-/second-semester academic variables from a Portuguese higher-education institution. The original dataset has 4,424 records; removing 794 Enrolled records leaves 3,630 observations: 2,209 Graduate and 1,421 Dropout. The final model uses **36 input features**.

Source: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. [DOI: 10.24432/C5MC89](https://doi.org/10.24432/C5MC89). Dataset license: CC BY 4.0. Raw student records are not bundled with this application.

## Data Preprocessing

The supplied `Phase3.ipynb` records checks for missing values and duplicate rows (both zero), duplicate removal, target filtering/encoding, removal of target columns from features, and numeric feature preparation. `pd.get_dummies(..., drop_first=True)` was called, but all 36 retained columns were already numeric.

**Important preprocessing finding:** the notebook first applied StandardScaler to the complete filtered dataset and exported `student_dropout_preprocessed.csv`. It subsequently loaded this already-standardized CSV, performed an 80/20 split, and fitted the saved second scaler on training rows only. Therefore, the saved scaler expects standardized CSV coordinates, not raw ages or grades.

The application reproduces this existing representation:

1. Copy `default_student.pkl`, whose values are already in CSV coordinates.
2. Convert the 12 raw interface values using fixed constants in `input_preprocessing.json`: `(raw_value - original_mean) / original_scale`.
3. Replace the corresponding default fields; preserve the other 24 defaults.
4. Construct all 36 columns in the exact order from `feature_columns.pkl`, including the original trailing tab in the attendance feature name.
5. Call the unchanged saved scaler's `transform`, then the unchanged model's `predict` and `predict_proba`.

The extra conversion constants were recovered from the original UCI dataset and verified against **every value in all 3,630 rows** of the supplied CSV (maximum absolute error `1.7763568394002505e-15`). This restores the notebook's input coordinates; it does not retrain, replace, or modify an estimator. All four original `.pkl` files retain their SHA-256 hashes.

## Exploratory Data Analysis

The notebook investigated the dropout distribution, academic performance, numerical/categorical-coded variables, and feature relationships. It includes a class-distribution chart, first-semester grade comparison, and feature–target correlations. Semester approvals/grades and tuition status had strong associations with the target. These are associations rather than causal effects.

## Machine Learning Model

The saved model is **Logistic Regression** (`max_iter=1000`, scikit-learn 1.6.1). Logistic Regression is appropriate for binary classification because it maps a linear combination of features through a logistic function to class probabilities. Standardization makes differing numerical scales more comparable. A predicted probability is a model estimate, not a verified individual outcome or a guarantee of calibration.

## Training

The final notebook split is stratified, with `test_size=0.2` and `random_state=42`: **2,904 training rows and 726 test rows**. The **saved second StandardScaler** was fitted only on training data, and its transform was applied to the test data.

However, the earlier full-dataset standardization and full-dataset median defaults introduce test-set information into preprocessing. The reported metrics have this **data-leakage limitation** and should not be presented as a fully leakage-free validation. The saved artifacts have been preserved as requested. Any future correction would require a separately authorized training run; this deployment performs inference only.

## Evaluation

The following values come from the supplied Colab notebook and were independently reproduced locally using the saved model, the supplied preprocessed CSV, and the recorded split—**without fitting any model or scaler**. Precision, recall, and F1 refer to the positive Dropout class.

| Metric | Measured test result |
|---|---:|
| Accuracy | 94.21% |
| Precision | 93.21% |
| Recall | 91.90% |
| F1-score | 92.55% |
| ROC-AUC | 0.9733 |

Confusion matrix (rows = actual class; columns = predicted class):

| Actual / Predicted | Not Dropout | Dropout |
|---|---:|---:|
| Not Dropout | 423 | 19 |
| Dropout | 23 | 261 |

A **false positive** is a Not Dropout student predicted as Dropout (19 cases). A **false negative** is a student belonging to the Dropout class predicted as Not Dropout (23 cases): these students may be missed by a support-identification process. See `evaluation.json` for unrounded reproduced values. These historical metrics describe the complete 36-feature test rows, not the accuracy of using 24 defaults in this simplified interface.

## Student Risk Prediction Application

The interface exposes age, admission/previous grades, tuition/debtor/scholarship flags, and enrolled/approved units plus average grades for each semester. Other variables come from the saved defaults. The complete representation is converted/scaled as documented above and passed to the trained model.

The displayed class comes from `model.predict()`. Dropout probability comes from `model.predict_proba()[:, 1]`, after verifying `model.classes_ == [0, 1]`.

| Probability | Risk category |
|---|---|
| Below 30% | LOW RISK |
| At least 30% and below 60% | MEDIUM RISK |
| At least 60% | HIGH RISK |

**These thresholds are application-defined interpretation bands, not thresholds learned by Logistic Regression.** A medium-risk result can have either predicted class. Bands use the unrounded probability; the percentage display is rounded to two decimal places.

Validation rejects missing/non-finite numbers, negative grades/counts, fractional ages/counts, and approved units exceeding enrolled units. Admission/previous grades use a 0–200 scale; semester grades use 0–20. Age 1–120 and unit counts 0–100 are broad application guardrails, not claims about observed training coverage. Actual observed feature minima/maxima are recorded in `input_preprocessing.json`.

## Example Predictions

These are **illustrative test cases evaluated with the saved artifacts**, not real students or measured outcomes. The remaining 24 features use the same saved defaults in every case.

| Input | Profile A | Profile B | Profile C |
|---|---:|---:|---:|
| Age | 20 | 23 | 25 |
| Admission grade | 150 | 125 | 110 |
| Previous grade | 145 | 125 | 110 |
| Tuition up to date | Yes | Yes | No |
| Debtor | No | No | Yes |
| Scholarship | Yes | No | No |
| First semester enrolled / approved / grade | 6 / 6 / 15 | 6 / 4 / 11 | 6 / 1 / 8 |
| Second semester enrolled / approved / grade | 6 / 6 / 15 | 6 / 3 / 10 | 6 / 0 / 0 |
| Dropout probability | 2.63% | 80.71% | 100.00% (rounded) |
| Predicted class | Not Dropout | Dropout | Dropout |
| Risk category | LOW RISK | HIGH RISK | HIGH RISK |

Profile C's unrounded probability is `0.9999764399708152`, not certainty. Expand **Try illustrative student profiles**, select an example, and click **Predict student risk**.

## Educational Early-Warning System

The system could help identify students who may require additional academic or financial support. It is a **prototype/educational project**. Predictions should support human decision-making rather than automatically determine actions about students.

The interface requires second-semester information, so it cannot serve as an enrollment-time predictor. Results may be affected by dataset/model limitations, class selection, saved defaults, the preprocessing leakage described above, and changes in student populations. Fairness and probability calibration have not been established. Hidden default values are not individualized observations. The app does not ask for names/IDs and does not write student inputs to a database.

## Project Structure

```text
student-dropout-risk-prediction/
├── app.py                         # Gradio interface
├── prediction.py                  # Validation, input conversion, saved-artifact inference
├── logistic_regression_model.pkl  # Original trained model
├── scaler.pkl                     # Original fitted scaler
├── feature_columns.pkl            # Original 36-feature order
├── default_student.pkl            # Original defaults
├── input_preprocessing.json       # Verified fixed raw-to-CSV coordinate conversion
├── artifact_checksums.json        # Original artifact SHA-256 hashes
├── evaluation.json                # Reproduced historical evaluation
├── requirements.txt
├── render.yaml                    # Free Render web-service configuration
├── tests/test_prediction.py
├── README.md
├── SCREENSHOTS.md
├── .gitattributes
└── .gitignore
```

## Installation

Use **Python 3.12**. Matching scikit-learn 1.6.1 is necessary for compatibility with the saved artifacts.

```bash
git clone https://github.com/Ahsan-Qamar-Dev/student-dropout-risk-prediction.git
cd student-dropout-risk-prediction
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS/Linux. Then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python app.py
```

Open `http://127.0.0.1:7860`. Only load trusted pickle files. The app verifies artifact checksums at startup and never accepts user-uploaded models.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Logistic Regression, Gradio, and Joblib.

## Deployment

Primary deployment target: a **free Render Python web service**. Hugging Face was checked first, but its current [Spaces policy](https://huggingface.co/docs/hub/spaces-overview) requires a paid account plan to create a Gradio compute Space. The application remains compatible with Hugging Face; the README YAML specifies Gradio 6.9.0, Python 3.12, and `app.py`.

For Render, create a Web Service from this public GitHub repository with Python 3.12. Use build command `pip install -r requirements.txt`, start command `python app.py`, and environment variable `GRADIO_SERVER_NAME=0.0.0.0`. Select the **Free** instance. `render.yaml` records the configuration. Render supplies `PORT`. No API key is required by the running application. Free services sleep after inactivity and can take time to restart; see [Render's free service limits](https://render.com/docs/free).

For a Hugging Face account with an eligible plan, create a public Gradio Space and upload all repository files, including the four `.pkl` artifacts and preprocessing/checksum JSON files. See the [Spaces configuration reference](https://huggingface.co/docs/hub/spaces-config-reference) and [Gradio quickstart](https://www.gradio.app/guides/quickstart).

**Current public deployment status:** the free ($0/month) Render service is configured, but creation is blocked by the account's card-verification prompt. Render states that verification uses a temporary $1 authorization hold. No live application is claimed yet. A live URL will be added only after the public application loads and successfully produces a prediction.

Local validation passed: all seven automated tests, startup and artifact loading, three HTTP API predictions, invalid-input rejection, and a browser prediction (2.63%, Not Dropout, LOW RISK). All original artifact hashes match the supplied files.

See [SCREENSHOTS.md](SCREENSHOTS.md) for the four internship evidence captures.
