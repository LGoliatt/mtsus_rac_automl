# MTSUS-RAC-AutoML

This repository presents an **AutoML benchmarking framework** for predicting compressive strength (CS) and flexural strength (FS) of **Recycled Aggregate Concrete (RAC)** using a dataset published by Yuan et al. (2022). The code supports a comparison among multiple AutoML libraries including:

- [FLAML](https://github.com/microsoft/FLAML)
- [TPOT](https://github.com/EpistasisLab/tpot)
- [AutoGluon](https://github.com/autogluon/autogluon)
- [AutoKeras](https://github.com/keras-team/autokeras)
- [H2O AutoML](https://github.com/h2oai/h2o-3)
- *(optionally)* AutoSklearn

## 📊 Objective

To evaluate and compare the performance and permutation-based feature importance of various AutoML frameworks on a real-world regression task involving sustainable construction materials.

## 📁 Repository Structure

```
├── data/
│   └── data_yuan/
│       └── yuan2022_recycled_aggregate_concrete.txt
├── rca_automl_comparison_cleaned.py   # Main experiment loop
├── read_data_rca.py                   # Dataset loading and preprocessing
└── README.md
```

## 📦 Requirements

Install the following libraries (using `pip`, `conda`, or environment manager):

```bash
pip install numpy pandas seaborn matplotlib scikit-learn flaml tpot autokeras autogluon h2o
```

⚠️ Note: AutoKeras requires TensorFlow and GPU support for best performance.

You must also have:

- Python ≥ 3.7
- Linux or macOS (Windows support may require Docker or WSL)
- LaTeX (for compiling statistics tables, optional)

## 🚀 How to Run

1. **Download the dataset** and place `yuan2022_recycled_aggregate_concrete.txt` under `./data/data_yuan/`.
2. Run the experiment:

```bash
python rca_automl_comparison_v_0p1.py
```

3. JSON outputs will be saved per run and per target (CS or FS) under folders like `./json_automl_cs/`.

## 📈 Outputs

Each run stores:
- Predictions (`y_test`, `y_pred`)
- R², MAE, RMSE, MSE scores
- Permutation feature importance values
- Metadata including time, seed, and method

All saved in `.json` format for reproducibility and visualization.

## 📚 Dataset Reference

Yuan, Y. et al. (2022).  
**Machine Learning Prediction of Mechanical Properties of Recycled Aggregate Concrete**  
*Materials*, 15(8), 2823.  
📎 [https://www.mdpi.com/1996-1944/15/8/2823](https://www.mdpi.com/1996-1944/15/8/2823)

## 🧠 Author & Contact

Developed by [@LGoliatt](https://github.com/LGoliatt) and collaborators for research in sustainable construction materials and AutoML benchmarking.

---

Feel free to fork, cite, and build upon this work.
