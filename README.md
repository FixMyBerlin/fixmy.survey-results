fixmycity survey results
==============================

Analysis of the Survey Results for FixMyCity
Author: Tümer Tosik

Project Organization
------------

    ├── LICENSE
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate and load data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 summary
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create visualizations
    │   │   └── visualize.py
    │   │
    │   └── exploration    <- Scripts to explore dataset
    └── README.md          <- The top-level README for developers using this project.

Reproducing Results
--------
1. Setup environment
```
	conda create --name fixmycity python=3.8
    conda activate fixmycity
```
2. Install requirements:
```
    python -m pip install -U pip setuptools wheel
    python -m pip install -r requirements.txt
```
3. Move raw datafiles into data/raw and run:
```
    python src/data/make_dataset.py
```
4. Train respective models:
```
    python src/models/train_model.py --experiment="<experiment name>"
```
You can choose between ["MS", "CP", "SE", "all"]

Notes:
 - Tr_li-Breite, Tr_re-Breite, RVA-Breite were replaced by their respective meter values


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
