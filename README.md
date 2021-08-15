# CHURN PREDICTION FINAL PROJECTS

## PROGRESS

Goal: 7-page Research Paper + 2-page executive summary

	1. Executive Summary: 2 pages
	2. Abstract: 0.5 pages
	3. Introduction: 1 page
	4. Literature Review: 2 pages
	5. Data and Methodologies: 1 page
	6. Results and Discussions: 3 pages
	7. Conclusion: 0.5 pages
	8. Reference/ Appendix: not count

Deadline: 

	- 18 Aug: Final paper presentation
	- 23 Aug: Final paper submission

Assigning Task + Timeline

| Task | Sub-task | Description | Assignment | Date | Expected Output |
|----|-------|----------|----------|----|-------------|
| Hypothesizing the problem | Research questions | Raising problem and ask research questions which aim to contribute the up-to-date scientific studies and afterwards conduct the models | Team | Aug 14 2021 | Several research questions and their contribution |
| Literature review | Collecting and Writing literature review | Focus the literature which discusses on our research problems, its pros and cons, and how our study overcomes/improves/differentiates their studies. We do at the same time with the team of data modeling to discuss together on how our study overcomes/improves/differentiates their studies | 2 persons | Aug 16 2021 | Essay |
| Exploratory data analysis | Univariate and Multivariate | Try to combine our knowledge in this course to conduct analyses and figure out some valuable insights. Those insights should use as a potential input for upcoming models | 2 persons | Aug 15 2021 | Essay |
| Data mining | After EDA, creating more useful variables and conduct several models | Try to code clean and clear | 2 persons from EDA | Aug 16 2021 | Output of model |
| Result and Discussion | Writing result and discussion | From the output, writing result | Team | Aug 17 2021 | Essay |
| Conclusion and Introduction | Writing conclusion and introduction | From above outputs | Team | Aug 17 2021 | Essay |
| Executive summary and Abstract | Writing executive summary and abstract | From above outputs | 1 person | Aug 17 2021 | Essay |
| Prepare PPT file for presentation | PPT preparing and Presenting | |2 persons, team answer the questions from Prof. | Aug 18 2021 | 1 file PPT and Presenting via Skype |
| Document for submission | Writing paper | Adjust paper from the comments in presentation, customize the paper such as format, reference, citation, etc. | Team | Aug 23 2021 | 1 file PDF |

Update Tool: GitHub for storing document and code files (Python and R), easy for sync and updating the info of code changes.

# Exploratory Data Analysis

Please see `EDA.html` for further details. Some highlights as follows.

- Imbalanced dataset: 84% existing customers
- Among categorical variables the percentage of Attrited Customers seems to be fairly equal across all categories of all the variables. `Gender` and `Income_Category` clearly contribute to discriminating power. Other categorical variables need check further.
- Detecting several continuous variables having large amount of outliers based on IQR rule. Majority of them follow non-Normal distribution. Several variables show remarkedly skewed to the right in their distributions. Some relationships show non-linearity. `Total_Relationship_Count`, `Months_Inactive_12_mon`, `Contacts_Count_12_mon`, `Credit_Limit`, `Total_Revolving_Bal`, `Total_Amt_Chng_Q4_Q1`, `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Avg_Utilization_Ratio` confirmedly have discrimination power.
- Drop `Avg_Open_To_Buy`
- Employ PCA for mixed data. Choosing 17 components for 71.5% of total variation. **Need further interpretation**.

# Data mining:

We employ logistic regression to describe several aspect of data. We will facilitate some techniques to handle the data and compare their performance in Logistic regression model to the benchmark. The considered techniques are as follows.

- Weight of Evidence transformation: aiming to handle non-linearity and outliers
- PCA transformation: aiming to handle multicolinearity
- SMOTE sampling: aiming to handle imbalance in the dataset
- Benchmark model: Logistic regression without aforementioned techniques.

The Performance criteria should be: the area under the receiver operating characteristic curve (AUC). The AUC assesses the behavior of a classifier disregarding class distribution, classification cutoff and misclassification costs (10.1016/j.ejor.2011.09.031)

Model can make two kinds of wrong predictions:

- Predicting that a customer will cancel their Credit Card services but doesnt : False Positive
- Predicting that a customer wont cancel their Credit Card servicebut does : False Negative

The bank's objective is to identify all potential Customer's who wish to close their Credit Card Services. Predicting that customers won't cancel their Card Serivces but they do end up attriting, will lead to loss. Hence the False Negative values must be reduced Metric for Optimization in final model to choose the best cutoff probability. The Recall must be maximized to ensure lesser chances of False Negatives.

Please see `Modeling.html` for further details. Some highlights are as follows.

- SMOTE does improve the benchmark performance in terms of both AUC and Recall.
- WOE improves the benchmark performance in terms of only Recall.
- PCA reduces the benchmark performance in terms of both AUC and Recall.
