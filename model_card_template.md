# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a logistic regression model to predict whether an individual earns more than $50,000 per year based on U.S. Census data. The model is built using scikit-learn and trained on a mix of demographic and employment-related features. Categorical features are handled using one-hot encoding, and the target variable is binarized before training. The trained model, along with the encoder and label binarizer, is saved to disk so it can be reused for evaluation and deployment through an API.


## Intended Use
The purpose of this model is to demonstrate how to build, evaluate, and deploy a complete machine learning pipeline. It is intended for learning and experimentation only. This model should not be used to make real-world decisions about a person’s income, employment opportunities, or access to services.


## Training Data
The model was trained using the UCI Adult Census Income dataset. This dataset includes demographic and work-related information such as age, education level, occupation, workclass, marital status, race, sex, and native country. The target label represents whether an individual’s income is greater than $50,000 per year. The data was split into training and testing sets using a stratified split to maintain class balance.


## Evaluation Data
Model performance was evaluated on a held-out test set taken from the same Census dataset. In addition to overall test performance, the model was evaluated on multiple slices of the data based on categorical features such as workclass, education, occupation, race, sex, and native country. These slice-level evaluations help highlight where the model performs well and where it struggles. The detailed results for each slice are saved in slice_output.txt.


## Metrics
_Please include the metrics used and your model's performance on those metrics._

The model was evaluated using precision, recall, and F1-score. On the overall test dataset, the model achieved a precision of approximately 0.74, a recall of approximately 0.56, and an F1-score of approximately 0.64. Slice-level analysis showed noticeable variation in performance across different groups. Categories with more data generally produced more stable results, while groups with very few samples often showed extreme or unreliable metric values.

## Ethical Considerations
This dataset contains sensitive attributes such as race, sex, marital status, and native country. Because the model learns patterns from historical data, it may reflect existing societal biases present in the dataset. As a result, predictions may be less accurate or fair for certain groups. This reinforces why the model should not be used for real-world decision-making that could impact individuals.


## Caveats and Recommendations
The model’s performance depends heavily on the quality and balance of the underlying data. Some data slices contain very small sample sizes, which makes their metrics unreliable. Future improvements could include collecting more balanced data, exploring alternative models, adding fairness-aware evaluation, and expanding testing around slice-based performance. Any real-world use would require much deeper validation and bias analysis before deployment.