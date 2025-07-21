College Data Analysis and Clustering Project.

This project analyzes a dataset of colleges based on various quality and location metrics. It includes data preprocessing, K-Means clustering, linear regression for fee prediction, and a variety of visualizations to explore patterns and trends in the data.

📊 Features Implemented
Data Preprocessing: Handles missing values and scales feature data for clustering.

K-Means Clustering: Groups colleges into clusters based on features like University Rating, Location Score, Infrastructure Score, and Faculty Quality.

Linear Regression: Predicts average college fees using selected input features.

Data Visualization:

Scatter plot of clusters by rating and fees

Boxplot showing fee distribution by cluster

Correlation heatmap of features

Histogram of average fees

Pie chart showing college distribution by cluster

Line chart of average fees trend over years

📁 Dataset
The dataset (college_data.csv) should contain the following columns:

University_Rating

Location_Score

Infrastructure_Score

Faculty_Quality

Average_Fees

(Optional) Year (for sales trend visualization)


📌 Requirements
Python 3.7+

pandas, numpy, matplotlib, seaborn, scikit-learn

📈 Output
The script will display multiple visualizations and print the predicted college fees for the test set.

