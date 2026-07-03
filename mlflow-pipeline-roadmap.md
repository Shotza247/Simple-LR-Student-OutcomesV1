# MLflow Pipeline Roadmap — Student Mark Prediction

## Overview

Extend `StudentMarkPrediction.ipynb` to:
1. Add a **Polynomial Regression** model alongside the existing Linear Regression
2. Wrap both models in **sklearn Pipelines**
3. Use **MLflow `autolog()`** to track experiments, compare model performance, and serve the UI over a locally hosted URL (accessible via browser, not just localhost on the machine)
4. Export structured **CSV artifacts** into 3 organised folders
5. Add a **pie chart** showing score-distribution percentages
6. Update the **architecture diagram**

---

## Sub-Tasks

---

### Sub-Task 1 — Folder Structure & CSV Exports

**Intent**  
Create 3 output folders inside the repo and write the correct CSVs into each one after data prep runs. This establishes the artifact store that MLflow will also log.

**Expected Outcomes**
- `data/original/` contains `Test_Score_Prediction_Data.csv` and `New_users_details.csv` (copies/symlinks)
- `data/processed/` contains `New_Test_Score_Data.csv`, `X_train.csv`, `X_test.csv`, `Y_train.csv`, `Y_test.csv`
- `data/predictions/` contains `predicted_test_scores.csv` (columns: `Hours_of_Study`, `Predicted_Test_Score`) written after each model run

**Todo List**
1. After `train_test_split`, add a cell that creates the 3 folders with `os.makedirs(..., exist_ok=True)`
2. Write `X_train`, `X_test` as CSVs into `data/processed/` with header `Hours_of_Study`
3. Write `y_train`, `y_test` as CSVs into `data/processed/` with header `Test_Score`
4. After each model's `predict()` call, write predictions to `data/predictions/<model_name>_predicted_test_scores.csv`
5. Copy (not move) the two original source files into `data/original/`

**Relevant Context**
- `X = df_final[['Hours_of_Study']].values` — shape `(50,1)`
- `y = df_final['Test_Score'].values` — shape `(50,)`
- Split: `train_test_split(X, y, test_size=0.2, random_state=42)`
- Source files: `Test_Score_Prediction_Data.csv`, `New_users_details.csv`

**Status** — `[x] done`

---

### Sub-Task 2 — Pie Chart Visualisation

**Intent**  
Add a pie chart cell that shows the four student segments as percentages of the full 50-student dataset.

**Expected Outcomes**
- A single pie chart with 4 slices:
  1. Above pass score (≥75) — students who passed the exam threshold
  2. Below pass score (<75) — students who failed the exam threshold
  3. Above pass rate (≥70% i.e. ≥`pass_rate`) — students above the computed class pass rate
  4. Below pass rate (<`pass_rate`) — students below the computed class pass rate
- Percentages displayed on slices; legend labelled clearly
- Chart saved as `data/processed/score_distribution.png` for MLflow logging

**Todo List**
1. After the existing histogram cell, add a new code cell
2. Compute the 4 segment counts using the already-defined `passed_students`, `failed_student`, `students_above_pass_rate`, `students_below_pass_rate` variables
3. Build labels, sizes, and colours lists; call `plt.pie()` with `autopct='%1.1f%%'`
4. Add title: `"Student Score Distribution"`
5. Save figure with `plt.savefig('data/processed/score_distribution.png')` before `plt.show()`

**Relevant Context**
- `pass_mark = 75`, `pass_rate` already computed as `np.sum(df_final['Test_Score'] >= 50) / len(df_final) * 100`
- Existing counts from notebook output: passed 18, failed 32, above_pass_rate 20, below_pass_rate 30

**Status** — `[x] done`

---

### Sub-Task 3 — sklearn Pipelines for Both Models

**Intent**  
Wrap the Linear Regression in a `Pipeline` and add a new Polynomial Regression pipeline. Both pipelines follow the same `fit → predict → evaluate` pattern, making them MLflow-comparable.

**Expected Outcomes**
- `linear_pipeline`: `Pipeline([('model', LinearRegression())])`
- `poly_pipeline`: `Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)), ('model', LinearRegression())])`
- Both trained on the same `X_train` / `y_train`; both evaluated on `X_test` / `y_test`
- Metrics printed side-by-side: MSE, RMSE, R², Accuracy for each

**Todo List**
1. Add `from sklearn.pipeline import Pipeline` and `from sklearn.preprocessing import PolynomialFeatures` to the imports cell
2. Replace the bare `LinearRegression()` training cell with a `linear_pipeline` version
3. Add a new cell immediately after for `poly_pipeline` with `degree=2`
4. Reuse existing metric variables (`mse`, `r2`, `rmse`, `accuracy_score`) for each pipeline separately — suffix variables `_lr` and `_poly`
5. Print a comparison table showing both sets of metrics

**Relevant Context**
- Current model cell: `model = LinearRegression()` → `model.fit(X_train, y_train)` → `y_pred = model.predict(X_test)`
- `PolynomialFeatures` must come before `LinearRegression` in the pipeline; it transforms the single `Hours_of_Study` feature into `[x, x²]`
- Keep `include_bias=False` — `LinearRegression` adds its own intercept

**Status** — `[x] done`

---

### Sub-Task 4 — MLflow `autolog()` & Experiment Tracking

**Intent**  
Instrument both pipeline runs with MLflow so all parameters, metrics, and artifacts are captured automatically. The tracking server must be configured to bind to `0.0.0.0` (not `127.0.0.1`) so the UI is reachable from a browser on the local network.

**Expected Outcomes**
- `mlflow.sklearn.autolog()` active before both pipeline `.fit()` calls
- Experiment name: `"Student-Score-Prediction"`
- Each pipeline run logged under a distinct run name: `"LinearRegression"` and `"PolynomialRegression-deg2"`
- MLflow tracking URI set to `http://0.0.0.0:5000` (bind address) — users browse to `http://<machine-ip>:5000`
- Artifacts logged: both model objects, `score_distribution.png`, and all CSVs from `data/`
- A startup instruction cell (Markdown) documents the command to launch the server: `mlflow ui --host 0.0.0.0 --port 5000`

**Todo List**
1. Add `import mlflow` and `import mlflow.sklearn` to the imports cell
2. Set tracking URI in a dedicated setup cell: `mlflow.set_tracking_uri("http://0.0.0.0:5000")`
3. Create/set experiment: `mlflow.set_experiment("Student-Score-Prediction")`
4. Wrap the `linear_pipeline` fit/predict/evaluate block in `with mlflow.start_run(run_name="LinearRegression"):`
5. Wrap the `poly_pipeline` block in `with mlflow.start_run(run_name="PolynomialRegression-deg2"):`
6. Inside each run block, after autolog, manually log remaining artifacts: `mlflow.log_artifact("data/processed/score_distribution.png")` and `mlflow.log_artifacts("data/")` 
7. Add a Markdown cell before the MLflow setup cell with the server launch command and network access note

**Relevant Context**
- `mlflow.sklearn.autolog()` captures: estimator params, fit time, MSE, R², model signature, input example
- `autolog()` must be called **before** `.fit()`, not after
- The tracking server itself is launched from terminal (`mlflow ui --host 0.0.0.0 --port 5000`), not from inside the notebook
- Network accessibility: `0.0.0.0` binds to all interfaces; the browser URL uses the machine's actual LAN IP, e.g. `http://192.168.x.x:5000`

**Status** — `[x] done`

---

### Sub-Task 5 — Architecture Diagram Update

**Intent**  
Replace / update `Architectural diagram flow.png` with a new diagram that reflects the full pipeline: dual models, MLflow tracking, folder outputs, and pie chart.

**Expected Outcomes**
- New Mermaid diagram (rendered to PNG or kept as `.md` source) covering the updated flow
- Diagram shows: data sources → merge → stats/EDA → pie chart → train/test split → CSV exports → Pipeline LR → Pipeline Poly → MLflow autolog → experiment UI

**Todo List**
1. Draft the Mermaid diagram in a new file `architecture-diagram-v2.md`
2. Add a reference to it in `README.md` under the existing Model Architecture section
3. Update the `README.md` "Model Architecture" section to mention both pipelines and MLflow

**Relevant Context**
- Existing diagram file: `Architectural diagram flow.png`
- Previous Mermaid diagram was produced in chat — the new one extends it with Poly pipeline, MLflow, and folder export branches

**Status** — `[x] done`

---

## Implementation Order

```
Sub-Task 1 (folders + CSVs)
    ↓
Sub-Task 2 (pie chart)
    ↓
Sub-Task 3 (pipelines)
    ↓
Sub-Task 4 (MLflow)
    ↓
Sub-Task 5 (diagram)
```

Each sub-task is independently reviewable. Sub-Tasks 3 and 4 are tightly coupled — confirm Sub-Task 3 is working before adding MLflow wrapping.

---

## Key Decisions to Confirm Before Implementation

1. **Polynomial degree** — plan uses `degree=2`; confirm or change
2. **Folder location** — plan uses `data/original/`, `data/processed/`, `data/predictions/` inside the repo root; confirm acceptable
3. **MLflow install** — `pip install mlflow` required; confirm it is or can be added to the env
4. **Server host** — `0.0.0.0:5000` makes the UI reachable on the LAN; confirm this is the intent (vs. a remote server)
5. **Pie chart segments** — 4 slices as described above; confirm segment definitions match your intent

---

*Plan file: `mlflow-pipeline-roadmap.md`*
