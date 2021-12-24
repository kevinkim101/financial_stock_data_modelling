Stock and Securities Data Modeling
==============================

The following is a passion project of two friends who share a mutual interest in fintech. We focus on the applications of various statistical and deep learning models and how well they work on predicting the price of stocks.

**The project is defined into 3 main parts:**
1) The notebooks, some rough work and notes done during the planning and prototyping stages of the project.
2) The models, this entails the code to generate the models, not the actual models themselves.
3) The visualization and analysis of the models and their performances.


The data used is courtesy of: https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs


Project Organization
------------

    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └|


--------


Purpose + What we Learned
------------
**Brian:** During my first co-op as an applied machine learning intern, I learned that none of the work of ML is in the making of the model, all of it lies in the analysis and application. I had studied all of the theory and the math behind the models but always fell victim to analysis paralysis when handed a solo project. The purpose of this project is not to show off technicals, but to work on analysis, baselining and proof of concept. 

**Kevin:**


How to use 
----------- 

