Anastasiia Stepanova
aa.stepanova@innopolis.university
BS20-RO

## Repo structure

Specified in the assignment task :)

```
text-detoxification
├── README.md # The top-level README
│
├── data 
│   ├── external # Data from third party sources
│   ├── interim  # Intermediate data that has been transformed.
│   └── raw      # The original, immutable data
│
├── models       # Trained and serialized models, final checkpoints
│
├── notebooks    #  Jupyter notebooks. Naming convention is a number (for ordering),
│                   and a short delimited description, e.g.
│                   "1.0-initial-data-exporation.ipynb"            
│ 
├── references   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures  # Generated graphics and figures to be used in reporting
│
├── requirements.txt # The requirements file for reproducing the analysis environment, e.g.
│                      generated with pip freeze › requirements. txt'
└── src                 # Source code for use in this assignment
    │                 
    ├── data            # Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── models          # Scripts to train models and then use trained models to make predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │   
    └── visualization   # Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

## Prerequisites

- conda
- Python 3.11

I used Python 3.11.5 installed via conda, but any Python 3.11 version should work. Python 3.11 is recommended by PyTorch, which is the core framework used.

Also conda is the package manager recommended by PyTorch, because it contains not only Python packages, but also binary distributions of third-party software (most important of which is CUDA toolkit).

Shipped `requirements.txt` file should produce a working installation of `venv`, but you need to manually install third-party software if you want to get decent performance.

## How to run?

```bash
# Prepare and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# [OPTIONAL] Prepare dataset if you plan to train.
python ./src/data/make_dataset.py

# [OPTIONAL] Train the model if needed. Probably you don't want to do this, as this requires enormous amount of time.
python ./src/models/train_model.py

# Use the model!
python ./src/models/predict_model.py
```

