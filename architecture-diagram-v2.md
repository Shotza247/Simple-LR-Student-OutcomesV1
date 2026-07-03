# Architecture Diagram v2 — Student Mark Prediction Pipeline

```mermaid
flowchart TD
    A[Test_Score_Prediction_Data.csv] --> ORIG[data/original/]
    B[New_users_details.csv] --> ORIG

    A --> C
    B --> D[Synthetic Data Generation\nnp.random.seed-42 - 25 rows\nTest_Score = 2.5x + noise clipped 0-100]
    D --> E[pd.concat + Student IDs\ndf_extended - 25 new students]
    E --> C[Combined Dataset - df_final\npd.concat dfvOne + df_extended\n50 students - dropna\nSaved to New_Test_Score_Data.csv]

    C --> F[NumPy EDA\nmean - median - max - min - range - pass_rate]
    F --> G[Segmentation\npassed_students >=75\nfailed_student <75\nstudents_above_pass_rate\nstudents_below_pass_rate\nexam_top_scorers >=85]
    G --> H[Histogram\nPassed vs Failed Distribution]
    G --> PIE[Pie Chart - 4 segments\nAbove Pass Score >=75\nBelow Pass Score <75\nAbove Pass Rate >=pass_rate\nBelow Pass Rate <pass_rate\nSaved to data/processed/score_distribution.png]

    C --> I[Feature + Target Extraction\nX = Hours_of_Study shape 50x1\ny = Test_Score shape 50]
    I --> J[train_test_split\n80 pct train 40 rows\n20 pct test 10 rows\nrandom_state=42]

    J --> CSVEXP[data/processed/\nNew_Test_Score_Data.csv\nX_train.csv - X_test.csv\nY_train.csv - Y_test.csv]

    J --> MLSETUP[MLflow Setup\nautolog enabled\nexperiment: Student-Score-Prediction\ntracking URI: http://0.0.0.0:5000]

    MLSETUP --> LR[Linear Pipeline\nPipeline - LinearRegression\nrun name: LinearRegression]
    MLSETUP --> POLY[Polynomial Pipeline\nPipeline - PolynomialFeatures deg2 + LinearRegression\nrun name: PolynomialRegression-deg2]

    LR --> LRP[y_pred_lr\ndata/predictions/lr_predicted_test_scores.csv]
    POLY --> POLYP[y_pred_poly\ndata/predictions/poly_predicted_test_scores.csv]

    LRP --> EVAL[Model Comparison Table\nMSE - RMSE - R2 - Accuracy\nLinearRegression vs PolynomialRegression-deg2]
    POLYP --> EVAL

    LR --> MLF[MLflow autolog\nparams - metrics - model artifacts\nscore_distribution.png + all data/ CSVs]
    POLY --> MLF

    MLF --> UI[MLflow UI\nbrowser: http - LAN-IP - 5000\nCompare runs side-by-side]
```

## Server Launch

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Browse to `http://<your-LAN-IP>:5000` — find your IP with `ipconfig` (Windows) or `ip a` (Linux/Mac).

## Folder Structure

```
project/
├── data/
│   ├── original/
│   │   ├── Test_Score_Prediction_Data.csv
│   │   └── New_users_details.csv
│   ├── processed/
│   │   ├── New_Test_Score_Data.csv
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── Y_train.csv
│   │   ├── Y_test.csv
│   │   └── score_distribution.png
│   └── predictions/
│       ├── lr_predicted_test_scores.csv
│       └── poly_predicted_test_scores.csv
├── StudentMarkPrediction.ipynb
├── Test_Score_Prediction_Data.csv
├── New_users_details.csv
├── New_Test_Score_Data.csv
└── mlruns/   ← created automatically by MLflow
```
