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

## Pipeline comparison

The *AutoML* model presented best results, but the model using hyperparameter tuning was not a bad choice. The big advantage of using *AutoML* instead of *HyperDrive* was that *AutoML* is faster to get started with, and already presents the result in a zip file with the conda environment and models ready to deploy. That said, parameter tuning is faster to train, and a lot more flexible in what you can be tuned.

## Future work
In the *AutoML* model trained, neural networks were not considered, and could present better results. Neural networks take more time to train, and are more expensive. Since we are experimenting for educational purposes only, it was not considered necessary to use Deep Learning. Future work to improve the pipeline could consider these more advanced algorithms that could present better results, the steps would be:

* To enable complex models like deep learning at the AutoML configuration.
* To use a deep learning framework (such as Keras) instead of SKLearn on train.py
