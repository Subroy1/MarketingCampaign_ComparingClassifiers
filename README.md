## Comparing Classifiers for Bank Marketing Campaign

### 1. Introduction
This assignment focuses on comparing the performance of various classification models (K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines) to predict customer responses to a bank marketing campaign. The objective is to identify potential subscribers to long-term fixed deposits, enabling more efficient and targeted marketing efforts.


### 2. Data Overview
The dataset is sourced from the UCI Machine Learning repository, specifically related to bank marketing campaigns by a Portuguese banking institution. It contains information from 17 marketing campaigns conducted between May 2008 and November 2010. The primary goal is to predict the binary outcome `y` (renamed to `Response`), indicating whether a client subscribed to a term deposit (`yes`/`no`).


**Key Data Characteristics:**
*   **Total Contacts:** Originally 79,354, filtered to 41,188 entries. After cleaning duplicates, 41,176 unique entries remain.
*   **Class Imbalance:** The target variable `Response` is highly imbalanced, with approximately 88.7% 'no' (did not subscribe) and 11.3% 'yes' (subscribed).
* <img width="569" height="461" alt="image" src="https://github.com/user-attachments/assets/8f5b4295-27d2-4974-a95f-5c8d60bc996e" />

*   **Features:** A mix of numeric and categorical features describing bank client data, temporal data, and social and economic context attributes. For this analysis, only 'bank client data' features were primarily used to avoid data leakage (e.g., 'duration' of the call during telemarketing).


### 3. Problem Statement & Business Objective
The business objective is to increase the efficiency of direct marketing campaigns for long-term deposit subscriptions. By accurately predicting probable subscribers, the bank can prioritize resources, reduce unnecessary contacts, and maximize campaign success rates. This involves building a predictive model that can identify customers most likely to subscribe.


### 4. Feature Endgineering steps

1.  **Data Loading & Initial Inspection:** Data was primarily clean with only a dozen duplicates which were removed , no null values in dataset .
2.  **Target Variable Preparation:** The `y` column was renamed to `Response` and mapped to numerical values (0 for 'no', 1 for 'yes').
3.  **Feature Selection:** To build a realistic predictive model and avoid data leakage, only 'bank client data' features (`age`, `job`, `marital`, `education`, `default`, `housing`, `loan`) were selected. Economic indicators which have nothing to do with customers and campaign-specific temporal features were explicitly excluded to avoid data leakage and also the fact that duration would not be available before making a marketing call .
4.  **Train/Test Split:** The dataset was split into training (80%) and testing (20%) sets, using `stratify=y` to maintain the original class distribution in both sets, crucial for handling the class imbalance.
5.  **Preprocessing Pipeline:** ColumnTransformer was used for transforming :
    *   **Numeric Features (age):** Imputed with the median (though no missing values were present) and scaled using StandardScaler.
    *   **Categorical Features:** Imputed with a 'missing' constant (though no missing values were present) and encoded using OneHotEncoder(handle_unknown='ignore').
    * We used Imputers to ensure that unseen data can be accomodated with the same model. 
6.  **Class Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique) was incorporated into the imblearn.pipeline.Pipeline to oversample the minority class ('yes') only on the training data to enhance model's ability to learn from the minority class (no in this case mapped to 0).


### 5. Model Comparison & Hyperparameter Tuning

**Performance Metric:** F1-score was chosen as the primary evaluation metric due to the significant class imbalance. F1-score provides a balanced measure of precision and recall, ensuring the model effectively identifies positive cases without excessive false positives or negatives, which is critical for optimizing marketing efforts. ROC curve was used as a complementary metric and the AUC was also plotted to visually understand the comparisons across models .

1.  **Baseline Model:** A DummyClassifier with a "most_frequent" strategy was used to establish a baseline which any of our models should beat. 
2.  **Initial Models (Default Settings):** Logistic Regression, Decision Tree, KNN, and SVC were trained with default parameters using the preprocessing and SMOTE pipeline. Initial results showed:
    *   Logistic Regression: F1 ~ 0.25
    *   Decision Tree: F1 ~ 0.21
    *   KNN: F1 ~ 0.22
    *   SVC: F1 ~ 0.26 (with significantly longer training time of 6 minutes)
      
3.  **Hyperparameter Tuning with GridSearchCV & Stratified K-Fold:** Each model underwent hyperparameter tuning using `GridSearchCV` combined with StratifiedKFold (5 folds) on the training data. This ensures robust evaluation and selection of the best parameters based on F1-score.
    *   **Logistic Regression:** Tuned `C` and `solver`.
    *   **KNN:** Tuned `n_neighbors` and `weights`.
    *   **Decision Tree:** Tuned `max_depth` and `min_samples_split`.
    *   **SVM:** Due to its high computational cost, SVM tuning was performed on 10% subset of the data. Tuned `C`, `gamma`, and `kernel` (`linear`, `rbf`).

### 6. Key Findings & Interpretations

*   **Best Performing Model:** The **Tuned Decision Tree Classifier** (with `max_depth=5`, `min_samples_split=2`) emerged as the best model based on its cross-validation F1-score (0.26) and test F1-score (0.266). The Tuned SVM also performed similarly (Test F1 of 0.269), but on a smaller dataset, and at a much higher computational cost. 

*   **Targeted Campaign Strategy:** The model provides a list of `probable_customers` ranked by their `response_probability`. This allows the bank to focus its telemarketing efforts on customers with the highest likelihood of subscribing, optimizing resource allocation and campaign effectiveness.

### 7. Limitations
1.  **Limited Feature Set:** The F1-scores, while significantly better than the random baseline, are quite below par expectations specifcially due to the exclusion of highly predictive features (e.g., duration`, poutcome and some economic indicators).
2.  **F1-Score Interpretation:** An F1-score around 0.27 reflects the challenge of predicting a rare event with a constrained feature set. While it represents an improvement over random guessing, it highlights the difficulty of the problem.
3.  **Decision Threshold:** Further tuning of the classification decision threshold based on business objectives (e.g., maximizing precision vs. recall) could yield different performance outcomes. Otherwise , the default of 0.5 is used by the models.


### 8. Improvements & Future Work

1.  **Expanded Feature Engineering:** Explore creating new features from existing data, such as:
    *   **Past Campaign Interaction:** Some of the removed features can be used such as campaign and poutcome as they can be strong indicators of customer behaviour.
    *   **Family Size:** Deriving this from marital status to understand household dynamics - More family members might point to requirement of more savings hence possibility of opening term deposit increases.
    *   **Seasonal Trends:** Incorporating information about holiday seasons or specific months that historically show higher response rates.
    *   **Opening Channels:** Information about some clients who used mobile banking, internet banking could be inclined to manage deposits (open, close, modify term duration etc , can be a strong indicator for the model).
    *   **Client Portfolio Strength:** For existing customers, the amount of money held idle in accounts could indicate a higher likelihood of opening a deposit.

2.  **Ensemble Models:** Investigate more advanced models like Random Forests(Bagging) or Gradient Boosting that can handle imbalanced datasets better.


Link to Jupyter Notebook -https://github.com/Subroy1/MarketingCampaign_ComparingClassifiers/blob/main/prompt_III_Assignment17_1.ipynb

