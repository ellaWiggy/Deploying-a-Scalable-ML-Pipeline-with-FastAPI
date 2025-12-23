# Model Card - Inference in Census Data

## Model Details
    - Ella Wiggins built the model with default Random Forest Classifier hyperparameters in scikit-learn 1.5.1.

## Intended Use
    - The model's intended use is to predict whether a person's income exceeds $50K per year based on various factors in the census data. 

## Training Data
    - Training data was split from the main dataset with a random state of 42, based on the categorical features including: workclass, education, marital-status, occupation, relationship, race, sex, and native-country.

## Evaluation Data
    - Testing data were split from the main dataset with a random state of 42, based on the categorical features: workclass, education, marital-status, occupation, relationship, race, sex, and native-country.

## Metrics
    - The model has an F1 score of 0.6817, a recall of 0.6336, and a precision of 0.7378.

## Ethical Considerations
    - The data was obtained from a 1994 census database and may have become less accurate over time in predicting a person's income. No new information has been inferred or annotated to the data. 

## Caveats and Recommendations
    - Recommendations for this dataset would be to obtain data from 2020 or later to get a better estimate of whether a person makes over $50K, and to add new features, since there may be more factors today that did not exist then.  
