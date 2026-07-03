# Student Mark Prediction Model 📚
## Problem Statement 
### NB: Truth be told, there's actually a lot more variable go into passing a certification exam, and i will continously increment on this project as a result of that.

**The Challenge:**  
Many students struggle with exam preparation and ask themselves: *"How many hours do I need to study to pass my certification exam?"* or *"If I study more, will my test score improve significantly?"* 



Without data-driven insights, students often rely on guesswork, leading to either:
- **Over-studying** (wasted time and energy)
- **Under-studying** (failing the exam)
- **No clear feedback** on expected outcomes based on their study effort

**The Solution:**  
This machine learning model creates a **predictive relationship** between study hours and test scores, enabling students to:
✅ Understand the correlation between effort and performance  
✅ Set realistic study targets based on desired scores  
✅ Get personalized score predictions before taking exams  
✅ Make data-driven study plans  

---

## Project Overview

This project implements a **Simple Linear Regression Model** that predicts student test scores based on hours of study. The model analyzes historical exam data from 50 students and generates predictions for new learners.

### Model Type
- **Algorithm:** Linear Regression
- **Features:** Hours of Study (continuous variable)
- **Target:** Test Score (0-100)
- **Dataset Size:** 50 student records
- **Train-Test Split:** 80-20 -> 40-10 rows

---

## The Data

### Original Dataset Sample
Original baseline data containing 25 student records (see Test_Score_Prediction_Data.csv):

| Hours_of_Study | Test_Score |
|-----|----|
| 2.5 | 61 |
| 10.0 | 75 |
| 15.0 | 85 |
| 20.5 | 87 |
| 24.0 | 44 |
| 31.0 | 90 |
| ... | ... |

### Key Statistics (combined 50-student dataset)
```
Mean Test Score:      56.44%
Median Test Score:    60.00%
Max Test Score:       90.00%
Min Test Score:       0.00%
Exam Pass Rate:       70.00% (score ≥ 50)
Passing Threshold:    75 points
```

### Student Performance Breakdown
- **Passed Exam (≥75):** 18 students
- **Failed Exam (<75):** 32 students
- **Maximum Test Score of Passed Students:** 90.0%
- **Minimum Test Score of Passed Students:** 75.0%
- **Average Test Score of Passed Students:** 83.48%
- **Maximum Test Score of Failed Students:** 74.0%
- **Minimum Test Score of Failed Students:** 0.0%
- **Average Test Score of Failed Students:** 41.22%

---

## Model Performance

### Evaluation Metrics
```
Mean Squared Error (MSE):  367.55
R² Score:                  0.61
Root Mean Squared Error:   19.17
Accuracy Score:            61.00%
```

**What this means:**
- **R² = 0.61:** The model explains 61% of the variance in test scores
- **MSE = 367.55:** Average squared prediction error; RMSE of ~19.17 units from actual values
- **Accuracy = 61%:** The model correctly predicts outcomes 61% of the time

---

## Key Insights

### 1. Strong Positive Correlation
There is a **clear positive relationship** between study hours and test scores:
- Each additional hour of study correlates with ~2.5 point increase in test score
- Students studying 30+ hours consistently score 85-90+

### 2. Minimum Study Threshold
- To have a reasonable chance of passing (≥75): **Study at least 12-15 hours**
- Below 10 hours: High risk of failing

### 3. Optimal Study Zone
- **20-25 hours:** Consistently high scores (87-90+)
- **Diminishing returns:** Beyond 30 hours, scores plateau around 90

### 4. Pass Rate Distribution
```
Score Distribution:
  Failed (<75):    64% of students
  Passing (75+):   36% of students
  Class Exam Pass rate: 36% passed exam
```

---

## How to Use This Model

### 1. Make a Prediction
```python
# Example: Predict score for 20 hours of study
hours_studied = 20
predicted_score = model.predict([[hours_studied]])
print(f"Predicted score for {hours_studied} hours: {predicted_score[0]:.2f}")
# Output: Predicted score for 20 hours: 84.67
```

### 2. Plan Your Study Schedule
- **Target Score: 80?** → Study ~18 hours
- **Target Score: 85?** → Study ~22 hours
- **Target Score: 90?** → Study ~28 hours
- **Target Score: 95?** → Study ~32+ hours

### 3. Track Progress
Monitor your actual scores vs. predicted scores to identify:
- If you're on track
- If you need to adjust study methods
- Which topics need more focus

---

## Model Architecture

See [`architecture-diagram-v2.md`](architecture-diagram-v2.md) for the full pipeline diagram.

### Models
- **Linear Regression Pipeline:** `Pipeline([('model', LinearRegression())])`
- **Polynomial Regression Pipeline:** `Pipeline([('poly', PolynomialFeatures(degree=2)), ('model', LinearRegression())])`

### Processing
```
Linear:     y = mx + b
Polynomial: y = ax² + bx + c
  x = Hours of Study
  y = Predicted Test Score
```

### Experiment Tracking
- **MLflow `autolog()`** logs params, metrics, and model artifacts for both pipelines
- **Experiment name:** `Student-Score-Prediction`
- **UI:** `mlflow ui --host 0.0.0.0 --port 5000` → browse `http://<LAN-IP>:5000`

### Output
- **Predicted Test Score:** Continuous value (0-100 range)
- **Artifacts:** CSVs in `data/`, pie chart in `data/processed/`, models in `mlruns/`

---

## Visualizations

### Hours of Study vs Test Score Scatter Plot
```
This graph shows:
- Actual data points (green): Historical student records
- Predicted regression line (red): Model's best-fit line
- Strong positive linear trend visible
- Some natural variation (noise) in student performance
```

### Student Performance Distribution
```
Histogram showing:
- Red bars: Students who failed (<75)
- Green bars: Students who passed (≥75)
- Right-skewed distribution toward higher scores
- Majority of students are successful
```

---

## Dataset Description

### Files Included
1. **Test_Score_Prediction_Data.csv** - Original baseline data (25 records)
2. **New_users_details.csv** - Student details (name, email, phone)
3. **New_Test_Score_Data.csv** - Combined dataset (50+ records)
4. **StudentMarkPrediction.ipynb** - Complete model implementation

### Data Dictionary
| Column | Type | Description |
|---|---|---|
| Student_ID | Integer | Unique student identifier |
| Name | String | Student name |
| Email Address | String | Student contact email |
| Phone Number | String | Student contact phone |
| Hours_of_Study | Float | Total hours studied (0-35) |
| Test_Score | Float | Final exam score (0-100) |

---

## Model Limitations & Considerations

⚠️ **Important Notes:**
1. **Single Feature Model:** Only uses study hours; other factors not included:
   - Prior knowledge/foundation
   - Study quality (not just quantity)
   - Test anxiety
   - Subject difficulty
   - Sleep quality, nutrition, etc.

2. **Linear Assumption:** Assumes straight-line relationship; may not capture complex patterns

3. **Limited Sample:** Model trained on 50 students; more diverse data would improve reliability

4. **Prediction Range:** Best for 10-30 hours; extrapolation beyond this range is less reliable

---

## Future Enhancements

### 🚀 Planned Improvements:
- [ ] **Polynomial Regression:** Capture non-linear relationships
- [ ] **Multiple Features:** Add prior GPA, attendance, practice test scores
- [ ] **Regularization:** Ridge/Lasso regression for better generalization
- [ ] **Cross-Validation:** More robust performance metrics
- [ ] **Student Clustering:** Identify different learner profiles
- [ ] **Time-Series Analysis:** Track performance trends over multiple exams
- [ ] **Feature Engineering:** Study quality metrics, topic-specific preparation

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Execution
1. Open `StudentMarkPrediction.ipynb` in Jupyter Notebook or VS Code
2. Run cells sequentially to:
   - Load and explore data
   - Train the model
   - Generate predictions
   - Visualize results
3. Make predictions for new study hours

---

## Key Takeaways

| Finding | Insight |
|---|---|
| **Correlation** | Strong positive: More study = Higher scores |
| **Pass Threshold** | 12-15 hours minimum for 75+ score |
| **Optimal Study** | 20-25 hours for consistent 85+ performance |
| **Model Accuracy** | R² 0.61 — reasonable for a single-feature model |
| **Next Step** | Use for study planning; combine with quality prep |

---

## References & Concepts

### ML Concepts Used
- **Linear Regression:** Simple supervised learning algorithm
- **Train-Test Split:** 80-20 validation methodology
- **Mean Squared Error:** Measures average prediction error
- **R² Score:** Measures goodness of fit (0 to 1 scale)

### Use Case Background
This project demonstrates:
- Real-world ML application in education
- Data preprocessing and exploration
- Model training and evaluation
- Predictive analytics for decision-making

---

## Contact & Questions

For questions about this model or dataset:
- Review the `Student_Grade-UseCase.txt` for detailed use case documentation
- Check `StudentMarkPrediction.ipynb` for full code implementation
- Analyze the generated visualizations for data patterns

---

**Last Updated:** January 2026  
**Model Status:** ✅ Active & Ready for Predictions  
**Data Version:** v2.0 (50+ student records)

---

*"Study smart, not just hard. Let data guide your preparation strategy."* 📊
