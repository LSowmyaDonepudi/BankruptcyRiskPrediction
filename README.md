# BankruptcyRiskPrediction

# Overview-
The financial market is affected by company and enterprise bankruptcy on several
fronts, so the necessity to predict insolvency among organizations by monitoring
multiple characteristics becomes even more critical. A greater understanding of
bankruptcy and the ability to foresee it will impact lending institution profitability
around the world. To provide such an assessment and to find the future possibilities
for a company’s bankruptcies, we have used the machine learning algorithm to
classify the chance and find the key features that have played an essential role in the
companies that have gone bankrupt.

# Introduction-

The bank datasets allow us to perform an analysis and modelling, giving us insights
into the company's status on bankruptcy.
The above dataset contains information on the likelihood of a Polish company going
bankrupt. We gathered the data from Developing Markets Information Service
(EMIS), a global information collection on emerging markets. Companies that went
bankrupt were studied from 2000 to 2012, while those still functioning were assessed
from 2007 to 2013.
Five categorization instances are identified based on the acquired data and the
forecasted period:
1. 1stYear: The data includes financial rates from the first year of the forecasting
period and a class label that indicates whether the company would go bankrupt after
five years. 7027 occurrences (financial statements) are included in the data, with 271
representing bankrupted enterprises and 6756 firms that did not go bankrupt
throughout the predicted period.
2. 2ndYear comprises financial rates from the forecasting period's second year
and a class designation that denotes bankruptcy status after four years. There are
10173 cases (financial statements) in the data, 400 of which are bankrupted
corporations and 9773 that did not go bankrupt during the predicted period.
3. 3rdYear: The data includes financial rates from the third year of the forecasted
period and a class label that shows whether the company is bankrupt after three
years. There are 10503 occurrences (financial statements) in all, with 495
representing bankrupted enterprises and 10008 companies that did not go bankrupt
throughout the projection period.
4. 4thYear provides financial rates from the forecasting period's fourth year and
a class designation that denotes bankruptcy status after two years. There are 9792
occurrences (financial statements) in the collection, with 515 being failed enterprises.
9277 businesses did not go bankrupt throughout the predicted period.
5. 5thYear provides financial rates from the 5th year of the forecasted period and
a class label indicating bankruptcy status after one year. There are 5910 instances
(financial statements) in the data, with 410 representing insolvent companies and
5500 representing enterprises that did not go bankrupt throughout the projected
period.


## Models
Two main predictive models are used:
1. **Logistic Regression**
   - Implemented with and without SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
2. **Naive Bayes**
   - A probabilistic model that calculates the likelihood of bankruptcy based on the distribution of financial ratios.
  
## Dependencies
This project requires the following Python libraries:
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn

## Discussion-
• For this dataset, recall is a better performance metric as there is no cost in falsely
identifying a non-bankrupt company as bankrupt but wrongly identifying a bankrupt
company as non-bankrupt can have some serious consequences.
• After implementing oversampling using SMOTE and using stratified k-fold, we can
see the improvement in the models especially in recall.
• Logistic Regression with SMOTE was the most balanced and well performing model
overall.
• On the other hand, Naïve bayes with stratified k-fold and SMOTE, had better
accuracy and recall out of all the variants of both models.
• The performance of the models could have been better if more data on bankrupt
company was available.
• For future reference, models used for anomaly detection might perform well on this
dataset since the dataset is highly imbalanced.
