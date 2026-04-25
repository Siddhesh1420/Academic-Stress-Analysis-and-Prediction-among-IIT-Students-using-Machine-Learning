# Academic Stress Among IIT Students

A primary survey-based data analytics project examining stress patterns among 102 IIT students — covering the complete workflow from data collection to a deployed web application.

## What's inside

| File | Description |
|---|---|
| `data_clean.ipynb` | Raw data cleaning and preprocessing |
| `eda.ipynb` | Exploratory data analysis — 10 structured questions |
| `model.ipynb` | Model building and comparison |
| `app.py` | Interactive Streamlit dashboard |

## Results

| Model | Accuracy |
|---|---|
| Baseline | 47.6% |
| Logistic Regression | **71.4%** |
| Random Forest | 66.7% |
| Decision Tree | 61.9% |

**Key finding** — 41% of IIT students report High stress. GAD score and sleep hours are the strongest predictors.

## Stack
`Python` `Pandas` `Scikit-learn` `Plotly` `Streamlit`

## Live App
ykfes2czkq9apprqwsnyq8e.streamlit.app

## Run locally
```bash
pip install requirements.txt
streamlit run Streamlit/app.py
```

