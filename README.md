# Diabetes Prediction
![pexels-n-voitkevich-6942082](https://github.com/user-attachments/assets/7de1604d-4dc3-4bf0-a4b7-2280c0299524)

## Project Overview

The aim of this project is to diagnostically predict whether a patient has diabetes
based on certain diagnostic measurements included in the dataset. In particular, all patients here are females
at least 21 years old of Pima Indian heritage.

## About Dataset 

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. Several constraints were placed on the selection of these instances from a larger database.  From the data set in the (diabetes.csv) file, we can find several variables, some of them are independent (several medical predictor variables) and only one target dependent variable (Outcome).

## Tools
- Python Libraries -numpy, pandas, matplotlib, seaborn and sklearn were used for data manipulation, 
visualization and building of the machine learning model.

## Data Cleaning/Preparation
- The columns—glucose, blood pressure, skin thickness, insulin, and BMI— had a minimum value of zero. As this indicates potential missing values, I addressed this by replacing these zero values with appropriate substitutes, such as the mean or median.
- There were no duplicate records.

## Exploratory Data Analysis (EDA)

During the EDA process, I examined the sales data to answer questions like:
- What is the distribution of diabetes outcomes (e.g., positive vs. negative)?
- How do various features (like age, BMI, glucose levels) correlate with diabetes outcomes?
- How does the insulin level vary among different age groups?
- What is the impact of age on the likelihood of diabetes diagnosis?

## Data Analysis

```python
# checking for the distribution of the data using a histogram plot
p= diabetes.hist(figsize = (20,20))
```
```python
# plotting a boxplot to detect outliers within the columns
plt.figure(figsize=(12,12))
for i,col in enumerate(['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'bloodpressure']):
    plt.subplot(3,3, i+1)
    sns.boxplot(x = col, data = diabetes)
plt.show()
```
You can find more information about the codes used on this notebook, [here](https://github.com/praise-aiwansoba/Diabetes-Prediction/blob/main/Diabetes%20Prediction.ipynb))

## Model Development

- The data was converted to a uniform scale using the standard scalar
- The features and target were split into train and test variables using the train_test_split
- The RandomForest Classifier and KNearestNeighbor Classifiers were used to develop the model
- I performed hyperparameter tuning using GridSearchCV to get the best parameters for the knn model

## Model Evaluation

The models were evaluated based on their accuracy, precision, recall and f1 score
### RandomForest Classifier
- Accuracy: 75%
- Precision: 81% for class 0 and 65% for class 1
- Recall: 81% for class 0 and 65% for class 1
- F1 Score: 81% for class 0 and 65% for class 1
- ROC_AUC Score: 73%

### KNearestNeighbor Classifier
- Accuracy: 74%
- Precision: 77% for class 0 and 66% for class 1
- Recall:  85% for class 0 and 54% for class 1
- F1 Score: 81% for class 0 and 59% for class 1
- ROC_AUC Score: 69%
  
Overall, the Random forest classifier had the best predictions.

## Insights/Findings
Here are some insights and findings that was uncovered:

1. **Prevalence of Diabetes:**
   - The percentage of individuals diagnosed with diabetes versus those without, highlighting the overall prevalence in the dataset.
2. **Distribution of Key Features:**
   - Glucose levels, BMI, age, and insulin levels often show distinct distributions, with specific ranges indicating higher diabetes risk.
3. **Correlation Analysis:**
   - Strong correlations between certain features, such as glucose levels and diabetes outcomes, may be observed, indicating important relationships.
4. **Impact of BMI:**
   - Higher BMI values tend to correlate with an increased likelihood of diabetes, reinforcing the link between obesity and diabetes risk.
5. **Age Factor:**
   - Older individuals may have a higher incidence of diabetes, showcasing the importance of age as a risk factor.
6. **Gender Differences:**
   - Differences in diabetes prevalence or related measurements between genders might emerge, suggesting varying risk factors or health behaviors.
7. **Skin Thickness and Insulin:**
   - Relationships between skin thickness and insulin levels may highlight metabolic health indicators relevant to diabetes risk.
8. **Outliers:**
   - The presence of outliers in features like insulin levels or glucose can indicate measurement errors or unique cases that may require further investigation.
  
## Recommendations

1. **Regular Screening:**
   - Implement routine screening for diabetes, particularly for individuals over a certain age or those with higher BMI, to catch potential cases early.
2. **Health Education Programs:**
   - Develop targeted education programs focusing on nutrition and physical activity to help at-risk populations understand the importance of maintaining a healthy lifestyle.
3. **Weight Management Initiatives:**
   - Promote weight management strategies, including access to dieticians and exercise programs, to help individuals lower their BMI and reduce diabetes risk.
4. **Monitoring High-Risk Groups:**
   - Focus healthcare resources on monitoring and supporting individuals with high glucose levels, elevated BMI, or high insulin levels.
5. **Personalized Care:**
   - Encourage healthcare providers to consider personalized treatment plans that take into account individual risk factors such as age, gender, and BMI.
6. **Community Health Initiatives:**
   - Launch community health initiatives that offer screenings and lifestyle coaching to encourage healthier habits among residents.
7. **Follow-Up Programs:**
   - Establish follow-up programs for individuals diagnosed with prediabetes to help prevent progression to diabetes.

## Limitations
I had to replace the zero values in the the following columns- glucose, blood pressure, skin thickness, insulin, 
and BMI with suitable values inorder to ensure a complete dataset for building the machine learning model. There were a 
few outliers but even then we can still see positive correlation between some features and the diabetes outcome.

## Conclusion
By developing this predictive models using the dataset, we can identify individuals at high risk for diabetes, enabling proactive interventions.
   
