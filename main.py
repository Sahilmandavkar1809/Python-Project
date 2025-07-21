import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
def load_data():
    return pd.read_csv('college_data.csv')

# Step 2: Preprocess data
def preprocess_data(data):
    data.dropna(inplace=True)
    features = ['University_Rating', 'Location_Score', 'Infrastructure_Score', 'Faculty_Quality']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    return data, data_scaled

# Step 3: K-Means Clustering
def apply_clustering(data, data_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data_scaled)
    return data

# Step 4: Scatter plot for clusters
def visualize_clusters(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='University_Rating', y='Average_Fees', hue='Cluster', data=data, palette='Set2')
    plt.title('College Clusters Based on Rating and Fees')
    plt.xlabel('University Rating')
    plt.ylabel('Average Fees')
    plt.show()

# Step 5: Predict fees using regression
def predict_fees(data):
    features = ['University_Rating', 'Location_Score', 'Infrastructure_Score', 'Faculty_Quality']
    X = data[features]
    y = data['Average_Fees']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nPredicted Fees for Test Data:")
    print(np.round(predictions, 2))

# Step 6: Boxplot of fees per cluster
def plot_boxplot(data):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Cluster', y='Average_Fees', data=data, palette='Set3')
    plt.title('Average Fees Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Fees')
    plt.show()

# Step 7: Heatmap of feature correlations
def plot_correlation_heatmap(data):
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.drop('Cluster', axis=1).corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()

# Step 8: Histogram of average fees
def plot_histogram(data):
    plt.figure(figsize=(8, 5))
    plt.hist(data['Average_Fees'], bins=10, color='skyblue', edgecolor='black')
    plt.title('Histogram of Average College Fees')
    plt.xlabel('Average Fees')
    plt.ylabel('Frequency')
    plt.show()

# Step 9: Pie chart of colleges per cluster
def plot_cluster_distribution(data):
    cluster_counts = data['Cluster'].value_counts()
    labels = [f'Cluster {i}' for i in cluster_counts.index]
    plt.figure(figsize=(7, 7))
    plt.pie(cluster_counts, labels=labels, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
    plt.title('Distribution of Colleges per Cluster')
    plt.show()

# Step 10: Sales trend (average fees over years)
def plot_sales_trend(data):
    if 'Year' not in data.columns:
        print("No 'Year' column found for sales trend plot.")
        return
    trend_data = data.groupby('Year')['Average_Fees'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.lineplot(x='Year', y='Average_Fees', data=trend_data, marker='o', color='green')
    plt.title('Average College Fees Trend Over Years')
    plt.xlabel('Year')
    plt.ylabel('Average Fees')
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == '__main__':
    data = load_data()
    data, data_scaled = preprocess_data(data)
    clustered_data = apply_clustering(data, data_scaled)
    visualize_clusters(clustered_data)
    predict_fees(clustered_data)
    plot_boxplot(clustered_data)
    plot_correlation_heatmap(clustered_data)
    plot_histogram(clustered_data)
    plot_cluster_distribution(clustered_data)
    plot_sales_trend(clustered_data)
