# Ridge, Lasso & Elastic Net Regression Practicals

This repository contains practical examples and a small app demonstrating linear-model regularization techniques: Ridge Regression, Lasso Regression, and Elastic Net. It includes a Jupyter notebook with exploratory data analysis and model experiments, the datasets used, and a simple Python application scaffold.

## Contents

- `Code/Krish_Ridge_Lasso.ipynb` — Jupyter notebook with data cleaning, EDA, model training (Ridge, Lasso, ElasticNet), hyperparameter search, evaluation and visualizations.
- `Dataset/Algerian_forest_fires_cleaned_dataset.csv` — cleaned dataset used in the notebook.
- `Dataset/Algerian_forest_fires_dataset_UPDATE.csv` — original/alternative dataset version.
- `application.py` — small Python app (lightweight runner; may expose a simple UI or an API depending on implementation).
- `Model/` — directory for saving trained model artifacts (check for joblib/pickle files here after training).
- `templates/` — contains `home.html` and `index.html` used by the app if it is a Flask-based app.

## Project overview

This collection is intended for hands-on learning and demonstration of:

- Differences between Ridge (L2) and Lasso (L1) penalties.
- Elastic Net as a convex combination of L1 and L2 penalties.
- Proper data preprocessing for regularized linear models (scaling, categorical handling).
- Model selection via cross-validation and simple hyperparameter search.

The included notebook walks through end-to-end steps: loading the dataset, cleaning and preprocessing, splitting into train/test, training Ridge/Lasso/ElasticNet models, evaluating metrics (MSE, MAE, R²), and plotting coefficient paths.

## Prerequisites

We recommend using Python 3.8+ and a virtual environment.

Common packages used in the notebook and app (example):

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter
- notebook
- joblib
- flask (only if `application.py` uses Flask)

Example `requirements.txt` entries (not provided by the repo — add/create if needed):

```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
joblib
flask
```

## Setup (Windows PowerShell)

Create and activate a virtual environment, then install dependencies:

```powershell
# create venv
python -m venv .venv
# activate (PowerShell)
.\.venv\Scripts\Activate.ps1
# install dependencies (if you created a requirements.txt)
pip install -r requirements.txt
# or install the common set
pip install numpy pandas scikit-learn matplotlib seaborn jupyter joblib flask
```

If you use Conda, create and activate an environment as you prefer.

## Running the notebook

Open the notebook with Jupyter Lab/Notebook and run the cells interactively:

```powershell
# from repository root
jupyter notebook Code\Krish_Ridge_Lasso.ipynb
```

The notebook contains the training pipeline and plots. It is the recommended way to reproduce the experiments and visualizations.

## Running the app (if available)

If `application.py` is a Flask app or similar, run it like this in PowerShell after activating the environment:

```powershell
python application.py
```

Visit http://127.0.0.1:5000/ in your browser (or the URL printed to the console). If the app expects an already-trained model, check the `Model/` folder for saved artifacts (e.g., `model.joblib`) or re-run the notebook to train and save one.

## How to reproduce trained model artifacts

1. Run the notebook and follow the training cells to train your preferred model.
2. Use joblib to save artifacts, for example:

```python
from joblib import dump
dump(best_model, 'Model/best_model.joblib')
```

3. Place the saved model file in the `Model/` folder so the app can load it.

## Notes & Best practices

- Always scale features before applying Ridge/Lasso/ElasticNet (e.g., StandardScaler).
- Lasso can zero coefficients (feature selection). Ridge shrinks coefficients but rarely zeros them.
- Elastic Net mixes both behaviours — useful when there are correlated features.
- Use cross-validation (GridSearchCV or ElasticNetCV/RidgeCV/LassoCV) for hyperparameter tuning.

## Project structure (quick reference)

```
.
├─ application.py
├─ README.md
├─ Code/
│  └─ Krish_Ridge_Lasso.ipynb
├─ Dataset/
│  ├─ Algerian_forest_fires_cleaned_dataset.csv
│  └─ Algerian_forest_fires_dataset_UPDATE.csv
├─ Model/
└─ templates/
   ├─ home.html
   └─ index.html
```

## Troubleshooting

- If imports fail, ensure the virtual env is activated and dependencies are installed.
- If notebook kernel errors appear, restart the kernel and re-run cells from top.
- If the app can't load a model, confirm the path and filename under `Model/` and that the model was saved with the same library versions (scikit-learn major versions can introduce incompatibilities).

## License & Contact

This repository is provided for educational purposes. Check the project root for a license file if needed. For questions, open an Issue in the repository or contact the maintainer.

---

Happy experimenting — tweak the regularization strength and mixing parameters and see how the coefficients and predictions change!
