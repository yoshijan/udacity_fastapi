# Model Card

Model card for the udacity project *Deploying a Machine Learning Model on Heroku with FastAPI*

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Based on the US census data the model tries to predict the salary class of an individual. As model type a Linear
Regression is used

## Intended Use

Based on the US census data the model tries to predict the salary class of an individual

## Factors

The following features went into the data set:
age,workclass,fnlgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country
A detailed description of the data could be found here:
https://archive.ics.uci.edu/ml/datasets/census+income

## Metrics

For valedating the trained machine learning model precision, recall, and F1 are used:
On the test data set:
precision: 0.7155172413793104
recall: 0.2579241765071473
fbeta: 0.37916857012334393
## Evaluation Data

Model is evaluated on a separate test data set.

## Training Data

Test and data training data is split 20:80 from the original data set

## Quantitative Analyses

Model metrics are okay but could be further tuned (more hyperparamters, more advanced model, more training
data/features)

## Ethical Considerations

The data set public available and anonymised to a certain extent. With more detailed data the model could be improved
but to do so data privacy regulation needs to be considered

## Caveats and Recommendations

It is just a showcase model to learn more about MLOps.
