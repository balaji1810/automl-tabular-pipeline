# AutoML Exam - SS25 (Tabular Data)
This repo is for the exam assignment of the AutoML SS25 course at the university of Freiburg.


## Installation

To install the repository, first create an environment of your choice and activate it. 

You can change the Python version here to the version you prefer.

**Virtual Environment**

```bash
python3 -m venv automl-tabular-env
source automl-tabular-env/bin/activate
```

**Conda Environment**

Can also use `conda`, left to individual preference.

```bash
conda create -n automl-tabular-env python=3.11
conda activate automl-tabular-env
```

Then install the repository by running the following command:

```bash
pip install -e .
```

You can test that the installation was successful by running the following command:
```bash
python -c "import automl"
```

We place no restrictions on the Python version or libraries you use, but we recommend using Python 3.10 or higher.

## Code

* `download-datasets.py`: This script downloads the suggested training datasets.

* `run.py`: A script that loads in a downloaded dataset, trains an _AutoML-System_ and then generates predictions for
`X_test`, saving those predictions to a file. For the training datasets, you will also have access to `y_test` which
is present in the `./data` folder.

* `./src/automl`: This is a python package that will be installed above and contain the source code for the AutoML pipeline

## Data

### Practice datasets:
The following datasets are provided for practice purposes:

* bike_sharing_demand
* brazilian_houses 
* wine_quality
* superconductivity 
* yprop_4_1

You can download the practice data using:
```bash
python download-datasets.py
```

This will by default, download the data to the `/data` folder with the following structure.
The fold numbers `1, ..., n` refer to **outer folds**, meaning each can be treated as a separate dataset for training and validation. You can use the `--fold` argument to specify which fold you would like.

```bash
./data
├── bike_sharing_demand
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
│   ├── 2
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
│   ├── 3
    ...
├── wine_quality 
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
    ...
```

## Running an initial test
This will train a AutoML system and generate predictions for `X_test`:
```bash
python run.py --task bike_sharing_demand --seed 20 --output-path preds-20-bsd.npy
```

You are free to modify these files and command line arguments as you see fit.


## Reference performance

| Dataset | Test performance |
| -- | -- |
| bike_sharing_demand | 0.9457 |
| brazilian_houses | 0.9896 |
| superconductivity | 0.9311 |
| wine_quality | 0.4410 |
| yprop_4_1 | 0.0778 |

The scores listed are the R² values calculated using scikit-learn's `metrics.r2_score`.
