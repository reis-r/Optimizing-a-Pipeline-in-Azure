# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided *Scikit-learn* model.
This model is then compared to an *Azure AutoML* run.

## Summary
The dataset contains information about direct marketing campaigns of a bank institution. The objective is to predict whether the client will subscribe for a deposit (classification problem).

The metric used was accuracy. The model who performed best was from the algorithm *VotingEnsemble*, obtaining an overall accuracy of 0.9175.

## Scikit-learn Pipeline

The data is directly downloaded from a web address, in the *CSV* format, to an *Azure TabularDataset* object. The data is then converted to a *Pandas DataFrame* cleaned, and the labels split into a separated *Pandas Series*.

The data is then separated between test and train sets, with a proportion of 20% for the test, and the 80% left for training. The algorithm trained on the *Scikit-learn* pipeline is Logistic Regression, and the hyperparameters tuned are the *Regularization Strength* and the *Maximum Number of Iterations*.

The Parameter Sampler is the Random Sampler, which returns random values on a defined search space. This method is faster, but may not provide the best possible results, like the Grid Sampler. The Regularization Strength was configured with uniform sampling, which gives a value uniformly distributed between the minimum and maximum possible values. It's the most basic and safe parameter sampling method for continuous variables. The Maximum Number of Iterations will be a choice among values between 1 and 1000, with no advanced discrete hyperparameter distribution. We should not treat this variable as continuous.

The stopping policy choose was the *BanditPolicy*, which is a termination policy based on a slack factor and evaluation interval, it terminates when the accuracy of a run is not within the slack amount compared to the best performing run. It's a less conservative policy that will be sufficient for our studies.

## AutoML

The model returned from the *AutoML Pipeline* was a *VotingEnsemble* algorithm, which is an ensamble model created by *AutoML*. That means *AutoML* created it's own algorithm composed of other algorithms: *XGBoostClassifier*, *LightGBM*, and *SGD*, with different weights.

The JSON obtained from the run interface fully explains the model and it's weights.

```json
{
    "runId": "AutoML_7dd83831-c436-41dd-ad98-c1348563baae_52",
    "runUuid": "61288651-aa20-4d02-93c0-ecdead346104",
    "parentRunUuid": "ec1c3aeb-7d6a-4a16-a7d7-e3242273b2f0",
    "rootRunUuid": "ec1c3aeb-7d6a-4a16-a7d7-e3242273b2f0",
    "target": null,
    "status": "Completed",
    "parentRunId": "AutoML_7dd83831-c436-41dd-ad98-c1348563baae",
    "startTimeUtc": "2020-11-23T20:24:33.331Z",
    "endTimeUtc": "2020-11-23T20:25:37.907Z",
    "error": null,
    "warnings": null,
    "tags": {
        "ensembled_iterations": "[51, 40, 1, 0, 22, 39, 33, 28, 15]",
        "ensembled_algorithms": "['XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'SGD']",
        "ensemble_weights": "[0.08333333333333333, 0.08333333333333333, 0.16666666666666666, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.25, 0.08333333333333333, 0.08333333333333333]",
        "best_individual_pipeline_score": "0.9155083513493242",
        "best_individual_iteration": "51",
        "model_explanation": "True"
    },
    "properties": {
        "runTemplate": "automl_child",
        "pipeline_id": "__AutoML_Ensemble__",
        "pipeline_spec": "{\"pipeline_id\":\"__AutoML_Ensemble__\",\"objects\":[{\"module\":\"azureml.train.automl.ensemble\",\"class_name\":\"Ensemble\",\"spec_class\":\"sklearn\",\"param_args\":[],\"param_kwargs\":{\"automl_settings\":\"{'task_type':'classification','primary_metric':'accuracy','verbosity':20,'ensemble_iterations':15,'is_timeseries':False,'name':'quick-starts-ws-127597','compute_target':'local','subscription_id':'888519c8-2387-461a-aff3-b31b86e2438e','region':'southcentralus','spark_service':None}\",\"ensemble_run_id\":\"AutoML_7dd83831-c436-41dd-ad98-c1348563baae_52\",\"experiment_name\":null,\"workspace_name\":\"quick-starts-ws-127597\",\"subscription_id\":\"888519c8-2387-461a-aff3-b31b86e2438e\",\"resource_group_name\":\"aml-quickstarts-127597\"}}]}",
        "training_percent": "100",
        "predicted_cost": null,
        "iteration": "52",
        "_azureml.ComputeTargetType": "local",
        "_aml_system_scenario_identification": "Local.Child",
        "run_template": "automl_child",
        "run_preprocessor": "",
        "run_algorithm": "VotingEnsemble",
        "conda_env_data_location": "aml://artifact/ExperimentRun/dcid.AutoML_7dd83831-c436-41dd-ad98-c1348563baae_52/outputs/conda_env_v_1_0_0.yml",
        "model_data_location": "aml://artifact/ExperimentRun/dcid.AutoML_7dd83831-c436-41dd-ad98-c1348563baae_52/outputs/model.pkl",
        "model_size_on_disk": "1716374",
        "scoring_data_location": "aml://artifact/ExperimentRun/dcid.AutoML_7dd83831-c436-41dd-ad98-c1348563baae_52/outputs/scoring_file_v_1_0_0.py",
        "model_exp_support": "True",
        "pipeline_graph_version": "1.0.0",
        "model_name": "AutoML7dd83831c52",
        "staticProperties": "{}",
        "score": "0.9174809727682393",
        "run_properties": "classification_labels=None,\n                              estimators=[('51',\n                                           Pipeline(memory=None,\n                                                    steps=[('standardscalerwrapper',\n                                                            <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f6ec48a89e8>",
        "pipeline_script": "{\"pipeline_id\":\"__AutoML_Ensemble__\",\"objects\":[{\"module\":\"azureml.train.automl.ensemble\",\"class_name\":\"Ensemble\",\"spec_class\":\"sklearn\",\"param_args\":[],\"param_kwargs\":{\"automl_settings\":\"{'task_type':'classification','primary_metric':'accuracy','verbosity':20,'ensemble_iterations':15,'is_timeseries':False,'name':'quick-starts-ws-127597','compute_target':'local','subscription_id':'888519c8-2387-461a-aff3-b31b86e2438e','region':'southcentralus','spark_service':None}\",\"ensemble_run_id\":\"AutoML_7dd83831-c436-41dd-ad98-c1348563baae_52\",\"experiment_name\":null,\"workspace_name\":\"quick-starts-ws-127597\",\"subscription_id\":\"888519c8-2387-461a-aff3-b31b86e2438e\",\"resource_group_name\":\"aml-quickstarts-127597\"}}]}",
        "training_type": "MeanCrossValidation",
        "num_classes": "2",
        "framework": "sklearn",
        "fit_time": "46",
        "goal": "accuracy_max",
        "class_labels": "",
        "primary_metric": "accuracy",
        "errors": "{}",
        "fitted_pipeline": "Pipeline(memory=None,\n         steps=[('datatransformer',\n                 DataTransformer(enable_dnn=None, enable_feature_sweeping=None,\n                                 feature_sweeping_config=None,\n                                 feature_sweeping_timeout=None,\n                                 featurization_config=None, force_text_dnn=None,\n                                 is_cross_validation=None,\n                                 is_onnx_compatible=None, logger=None,\n                                 observer=None, task=None, working_dir=None)),\n                ('prefittedsoftvotingclassifier',...\n                                                                                                  learning_rate='constant',\n                                                                                                  loss='modified_huber',\n                                                                                                  max_iter=1000,\n                                                                                                  n_jobs=1,\n                                                                                                  penalty='l2',\n                                                                                                  power_t=0.2222222222222222,\n                                                                                                  random_state=None,\n                                                                                                  tol=0.0001))],\n                                                                     verbose=False))],\n                                               flatten_transform=None,\n                                               weights=[0.08333333333333333,\n                                                        0.08333333333333333,\n                                                        0.16666666666666666,\n                                                        0.08333333333333333,\n                                                        0.08333333333333333,\n                                                        0.08333333333333333,\n                                                        0.25,\n                                                        0.08333333333333333,\n                                                        0.08333333333333333]))],\n         verbose=False)",
        "friendly_errors": "{}",
        "onnx_model_resource": "{}",
        "error_code": "",
        "failure_reason": "",
        "feature_skus": "automatedml_sdk_guardrails",
        "dependencies_versions": "{\"azureml-widgets\": \"1.18.0\", \"azureml-train\": \"1.18.0\", \"azureml-train-restclients-hyperdrive\": \"1.18.0\", \"azureml-train-core\": \"1.18.0\", \"azureml-train-automl\": \"1.18.0\", \"azureml-train-automl-runtime\": \"1.18.0\", \"azureml-train-automl-client\": \"1.18.0\", \"azureml-tensorboard\": \"1.18.0\", \"azureml-telemetry\": \"1.18.0\", \"azureml-sdk\": \"1.18.0\", \"azureml-samples\": \"0+unknown\", \"azureml-pipeline\": \"1.18.0\", \"azureml-pipeline-steps\": \"1.18.0\", \"azureml-pipeline-core\": \"1.18.0\", \"azureml-opendatasets\": \"1.18.0\", \"azureml-model-management-sdk\": \"1.0.1b6.post1\", \"azureml-mlflow\": \"1.18.0\", \"azureml-interpret\": \"1.18.0\", \"azureml-explain-model\": \"1.18.0\", \"azureml-defaults\": \"1.18.0\", \"azureml-dataset-runtime\": \"1.18.0\", \"azureml-dataprep\": \"2.4.2\", \"azureml-dataprep-rslex\": \"1.2.2\", \"azureml-dataprep-native\": \"24.0.0\", \"azureml-datadrift\": \"1.18.0\", \"azureml-core\": \"1.18.0\", \"azureml-contrib-services\": \"1.18.0\", \"azureml-contrib-server\": \"1.18.0\", \"azureml-contrib-reinforcementlearning\": \"1.18.0\", \"azureml-contrib-pipeline-steps\": \"1.18.0\", \"azureml-contrib-notebook\": \"1.18.0\", \"azureml-contrib-interpret\": \"1.18.0\", \"azureml-contrib-gbdt\": \"1.18.0\", \"azureml-contrib-fairness\": \"1.18.0\", \"azureml-contrib-dataset\": \"1.18.0\", \"azureml-cli-common\": \"1.18.0\", \"azureml-automl-runtime\": \"1.18.0\", \"azureml-automl-core\": \"1.18.0\", \"azureml-accel-models\": \"1.18.0\"}",
        "num_cores": "4",
        "num_logical_cores": "4",
        "peak_memory_usage": "2880788",
        "vm_configuration": "Intel(R) Xeon(R) CPU E5-2673 v3 @ 2.40GHz",
        "core_hours": "0.009857065"
    },
    "inputDatasets": [],
    "outputDatasets": [],
    "runDefinition": null,
    "logFiles": {},
    "revision": 9
}
```

## Pipeline comparison

The *AutoML* model presented best results, but the model using hyperparameter tuning was not a bad choice. The big advantage of using *AutoML* instead of *HyperDrive* was that *AutoML* is faster to get started with, and already presents the result in a zip file with the conda environment and models ready to deploy. That said, parameter tuning is faster to train, and a lot more flexible in what you can be tuned.

## Future work
In the *AutoML* model trained, neural networks were not considered, and could present better results. Neural networks take more time to train, and are more expensive. Since we are experimenting for educational purposes only, it was not considered necessary to use Deep Learning. Future work to improve the pipeline could consider these more advanced algorithms that could present better results, the steps would be:

* To enable complex models like deep learning at the AutoML configuration.
* To use a deep learning framework (such as Keras) instead of SKLearn on train.py
