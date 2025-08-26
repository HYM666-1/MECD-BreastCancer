
Scripts and raw feature data for manuscript "Multidimensional Cell-Free DNA Fragmentomics Enables Early Detection of Breast Cancer"



# GSML Introduction

## I.Project Background

GSML is a machine learning framework primarily designed for internal MECD projects within the company.

## II、Installation

No installation required.

## III.Usage

### Command-line tool for H2O AutoML training and model performance statistics.

#### h2o automl训练

> ./gsml  TrainH2oAutoML

Use the H2O AutoML module to train and predict on datasets and features, and save the relevant base models.

+ > **Parameters**
    + --model_id: Model ID (also used as the prefix for output files). [required]
    + --d_output： Output file path. [required]
    + --feature： Feature file path(s). [required] [multiple]
    + --train_dataset: Training dataset path (info.list). [required] [multiple]
    + --pred_dataset: Prediction dataset path (info.list). [required] [multiple]
    + --leaderboard_dataset: Dataset path (info.list) used to guide training. [required] [multiple]
    + --nthreads： Maximum number of threads for H2O initialization. [default: 10]
    + --max_mem_size:  Maximum memory for H2O initialization. [default: 20000M]
    + --max_models: Maximum number of sub-models allowed for AutoML training. [default: 200]
    + --max_runtime_secs_per_model：Maximum training time (in seconds) for each sub-model. [default: 1800]
    + --max_runtime_secs：Maximum total runtime (in seconds) for the entire AutoML process. [default: 0 (unlimited)]
    + --nfolds：Number of cross-validation folds for a single model. [default: 5]
    + --seed：Random seed for model training. [default: -1]
    + --stopping_metric：Metric to determine iteration stopping. [default: aucpr] ["AUTO", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr", "lift_top_group", "misclassification", "mean_per_class_error", "r2"]
    + --sort_metric: Metric for model ranking.[default: aucpr] ["auc", "aucpr", "logloss", "mean_per_class_error", "rmse", "mse"]
    + --stopping_tolerance: Specify the relative tolerance for the metric-based stopping criterion to stop a grid search and the training of individual models within the AutoML run. [default: 0.001]

+ > **Examples**

    ```bash
    ./gsml TrainH2oAutoML \
        --model_id test \
        --d_output . \
        --feature cnv.csv \
        --train_dataset train.info.list \
        --pred_dataset valid.info.list \
    ```

#### model performance statistics

> ./gsml  ModelStat

Perform statistical analysis of the model's specific performance based on the prediction results, dataset, and optimization project information. This can be an instance of a GSML model or a prediction score file.

+ > **Parameters**

    + --f_model: Path to the GSML model. (Either f_modelor f_scoremust be specified)
    + --f_score：Path to the model prediction score file. (Must include SampleIDand Scorecolumns)
    + --dataset：Path to the dataset. (Format: dataset_name,dataset_path)
    + --optimize：Path to the optimization project file. (Format: project_name,project_path)
    + --conf：Configuration file for calculating combined scores. Required if combined score computation is needed.
    + --spec_list：List of different specificity (spec) cutoffs to evaluate. [multiple] [default: [0.9, 0.05, 0.98]]
    + --stat: Metrics to be calculated. [default: all] Options: ["auc", "score", "pred_classify", "combine_score", "performance"]
    + --d_output: Output directory for results.

+ > **Examples**

    ```bash
    ./gsml ModelStat \
        --f_model test.gsml \
        --dataset Train,Train.info.list \
        --dataset Valid,Valid.info.list \
        --conf gsml.test.yaml \
        --d_output .
    ```



#### Model Predict

> ./gsml Predict

Use an existing GSML model to predict a new dataset

+ > **Parameters**

    + --f_model: gsml model path。[required]
    + --feature: Path to the feature file of the samples to be predicted. [required]
    + --dataset: Path to the dataset file of the samples to be predicted. (If specified, only samples present in both the dataset and feature files will be used for prediction)
    + --nthreads： Maximum number of threads used for model prediction. [default: 5]
    + --max_mem_size：Maximum memory allocated for model prediction. [default: 20000M]. [default: 20000M]

#### Stack Model
> ./gsml pipe_combine_best_model
+ > **Parameters**

   + --d_model_list TEXT       Path of base model. [cnv,cnv/]  [required]
   + --d_output TEXT           path of result  [required]
   + --train_info TEXT         Path of train info. [Train,Train.info.list] [required]
   + --pred_info TEXT          Path of valid info. [Valid,Valid.info.list]
   + --feature_list TEXT       Path of features.  [required]
   + --threads INTEGER         threads.
   + --n_top_models TEXT       n_top_models.  [default: 2,3,4,5]
   + --stacked_algo TEXT       stacked_algo.  [default: mean, glm]
   + --stat_cols TEXT          columns name of stat by group
   + --required_features       all features
   + --help                    Show this message and exit.  
 


