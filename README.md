# Insurance Cost Prediction
 Insurance Cost Prediction based on health conditions and physical features

 ---

1. Case Study Introduction and Objective
2. 
1.1 Background and Overview
Before modeling and advanced analytics, insurance premiums were typically predicted using manual, rule-of-thumb methods based on basic demographic factors (age, gender, health conditions) and historical claims data. Actuaries used statistical tables, industry benchmarks, and general risk assessments to estimate premiums, often relying on broad assumptions rather than personalized data-driven insights.
Health insurance premium predictions using machine learning models are crucial for accurately assessing risk and setting appropriate premiums. By leveraging data such as age, medical history, lifestyle, and other factors, these models can help insurers predict an individual's future healthcare costs. This leads to more personalized and fair pricing, ensuring that individuals with higher health risks are charged premiums reflecting their needs, while those with lower risks aren't overcharged. Machine learning models also enhance operational efficiency by automating the pricing process, improving profitability, and enabling better risk management for insurers. Additionally, they ensure competitive and sustainable pricing in the market.

1.2 Problem Statement
Insurance companies need to accurately predict the cost of health insurance for individuals to set premiums appropriately. However, traditional methods of cost prediction often rely on broad actuarial tables and historical averages, which may not account for the nuanced differences among individuals. By leveraging machine learning techniques, insurers can predict more accurately the insurance costs tailored to individual profiles, leading to more competitive pricing and better risk management.

1.3 Need statement
The primary need for this project arises from the challenges insurers face in pricing policies accurately while remaining competitive in the market. Inaccurate predictions can lead to losses for insurers and unfairly high premiums for policyholders.
By implementing a machine learning model, insurers can:
Enhance Precision in Pricing: Use individual data points to determine premiums that reflect actual risk more closely than generic estimates.
Increase Competitiveness: Offer rates that are attractive to consumers while ensuring that the pricing is sustainable for the insurer.
Improve Customer Satisfaction: Fair and transparent pricing based on personal health data can increase trust and satisfaction among policyholders.
Enable Personalized Offerings: Create customized insurance packages based on predicted costs, which can cater more directly to the needs and preferences of individuals.
Risk Assessment: Insurers can use the model to refine their risk assessment processes, identifying key factors that influence costs most significantly.
Policy Development: The insights gained from the model can inform the development of new insurance products or adjustments to existing ones.
Strategic Decision Making: Predictive analytics can aid in broader strategic decisions, such as entering new markets or adjusting policy terms based on risk predictions.
Customer Engagement: Insights from the model can be used in customer engagement initiatives, such as personalized marketing and tailored advice for policyholders.

1.4 Column Profile
The Insurance dataset comprises the following 11 attributes:
Age: Numeric, ranging from 18 to 66 years.
Diabetes: Binary (0 or 1), where 1 indicates the presence of diabetes.
BloodPressureProblems: Binary (0 or 1), indicating the presence of blood pressure-related issues.
AnyTransplants: Binary (0 or 1), where 1 indicates the person has had a transplant.
AnyChronicDiseases: Binary (0 or 1), indicating the presence of any chronic diseases.
Height: Numeric, measured in centimeters, ranging from 145 cm to 188 cm.
Weight: Numeric, measured in kilograms, ranging from 51 kg to 132 kg.
KnownAllergies: Binary (0 or 1), where 1 indicates known allergies.
HistoryOfCancerInFamily: Binary (0 or 1), indicating a family history of cancer.
NumberOfMajorSurgeries: Numeric, counting the number of major surgeries, ranging from 0 to 3 surgeries.
PremiumPrice: Numeric, representing the premium price in currency, ranging from 15,000 to 40,000 (I am considering this as INR).

---

2. Exploratory Data Analysis (EDA) and Feature Engineering
EDA is done to understand data patterns, identify relationships, detect outliers, and prepare for further analysis or modeling.

2.1 Basic Data Exploration
- We observed the dataset and found out that there are 986 records and no null values in our dataset.
- Univariate analysis helps in understanding the distribution of data with respect to each feature we want to analyse. 
- Age is spread out between 18 and 66, with slightly higher distribution in 30's, 40's and 60's.
- We calculated BMI (Feature engineering) as a combination of Height and Weight features (Weight(kg)/Height(cm)²) and below is the BMI distribution.
- BMI distribution is slightly right skewed, spread out between 15 and 50, majority of people have BMI lesser than 30. 18.5 to 25 is an accepatable healthy range of BMI, anything above or below is considered underweight, overweight and obese.
- Premium Price ranges from 15000 to 40000, majority group of people are paying Premium price of around 21k, followed by 30k, and then 15k.
- Boxplots for the individual features are drawn, which show outliers in Weight, BMI and Premium Price columns.

Bivariate Analysis
We analysed Premium Price for various age and BMI categories and premium price difference between people who have serious health conditions and people who are healthy. 
Average Premium Price by Age and BMI categoriesPremium Price by health conditionsPremium Price is evidently greater for people having known health conditions than people who do not have health conditions.
As the number of surgeries people have had increases, Premium Price also increases.

Multivariate Analysis
In Multivariate Analysis, correlation calculation was done between all the numeric features of the dataset. 
Dataset CorrelationWe can clearly see that Premium Price is affected majorly by Age, followed by Transplants, Number of major surgeries and Chronic diseases. 

3. Hypothesis Testing
Hypothesis testing is done to evaluate assumptions or claims about a population using sample data, determining if there's enough evidence to support or reject them. It's crucial for making data-driven decisions in fields like research, business, and healthcare, ensuring that conclusions are statistically valid and not due to random chance.
- Used T-tests, ANOVA, Chi-square tests for hypotheiss testing.

4. Data Pre-processing - Handling Outliers, Scaling and encoding
We used IQR method to find outliers in the dataset and found that there were 16 outliers in weight, 6 outliers in Premium Price and 22 outliers in BMI. But considering the real world data, these values are actually very much possible. So, keeping the data as it is for our prediction. 
We dropped KnownAllergies since it did not have relevant effect on Premium Price. Using train_test_split, dataset was divided into train and test sets for model training and testing respectively. 
Scaling was done using StandardScaler( ) since for basic models such Linear Regression are sensitive to the scale of input features. When features have different scales, the model may disproportionately weight larger values, leading to biased coefficients and poor convergence during training.

5. Model Selection, Training and Tuning
We tried modeling data using Linear Regression , Polynomial Linear Regression, Decision tree Regressor, Random Forest Regressor, Gradient Boosting Regressor and found that Random Forest Regressor model fits out data best and accurate compared to other models. 

6. Model Evaluation
Models were evaluated based on R2 Score, Adjusted R2 Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE).
Random Forest Model Insights:
Strong Performance on Both Sets: The tuned random forest model performs very well on both the training (R² = 0.89) and test (R² = 0.81) data, with a relatively small drop in performance when moving from training to testing. This indicates good generalization ability.
Good Accuracy: Even on the test set, the model explains 81% of the variance, and the error metrics (MSE, RMSE, and MAPE) show that the model's predictions are relatively close to the actual values.
Overfitting Check: There is a slight performance drop between the training and test data (from R² of 0.89 to 0.81, and an increase in MSE/RMSE), but the drop is not large, which indicates that overfitting is minimal, especially for an ensemble model like random forest.

Feature Importance: 
Age, Transplants, Weight, Chronic Diseases affect Premium Price the most compared to other features

7. Model Deployment using Flask and Streamlit
The final phase of the Insurance Cost Prediction project involves deploying the developed machine learning model into a practical, user-friendly application. This application will serve as a web-based calculator for insurance agents or customers, enabling them to estimate insurance premiums based on individual data inputs. Deployment will be carried out through a simple Flask API or a Streamlit application, both of which are popular frameworks for deploying data science projects due to their ease of use and flexibility.
7.1 Model Deployment using Flask API Development
We created Flask application that serves as the backend, created endpoints that receive user inputs in the form of JSON and return the estimated premiums. Also, we integrated the trained machine learning model into the Flask application to process input data effectively. 

7.2 Model Deployment using Streamlit.io
We developed a User interface for customers to input their preferences and get Premium Price Predictions using Streamlit application. We used streamlit widgets and components and set up the application to use the model to predict premiums based on user inputs and display the results directly in the application.
When user inputs their data and clicks on Predict Premium, they are shown the Premium Price and Confidence Interval (95%) range. Along with it, they can also download their prediction report. 

8. Project Relevant links
Github link to access project: https://github.com/KeertiJoshi-9/Insurance-Cost-Prediction
Tableau Link for Data Visualization: https://public.tableau.com/app/profile/keerti.j1682/viz/InsuranceCostPredictionProject/InsuranceCostPrediction 
