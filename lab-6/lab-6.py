import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

data = {
    'Study_hours': [2, 4, 6, 8, 10],
    'Attendance': [60, 70, 80, 90, 100],
    'Previous_Marks': [55, 65, 75, 85, 95],
    'Assignment_Marks':[50,60,70,80,90]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)


scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)

normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
print("Normalised Data:")
print(normalized_df)

scaler = StandardScaler()
standardized_data = scaler.fit_transform(df)
standardized_df = pd.DataFrame(standardized_data, columns=df.columns)
print("Standardized Data:")
print(standardized_df)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

data = {
    'Study_hours': [2, 4, 6, 8, 10],
    'Attendance': [60, 70, 80, 90, 100],
    'Previous_Marks': [55, 65, 75, 85, 95],
    'Assignment_Marks':[50,60,70,80,90]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)


scaler = StandardScaler()
standardized_data = scaler.fit_transform(df)
standardized_df = pd.DataFrame(standardized_data, columns=df.columns)
print("Standardized Data:")
print(standardized_df.round(6))


covariance_matrix=np.cov(standardized_data,rowvar=False)
print("Covariance_Matrix")
print(covariance_matrix.round(6))



eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
eigen_values =eigen_values.real
eigen_vectors =eigen_vectors.real
print("Eigen Values:")
print(eigen_values.round(6))

print("Eigen Vectors:")
print(eigen_vectors.round(6))

index = np.argsort(eigen_values)[::-1]

eigen_values = eigen_values[index]
eigen_vectors = eigen_vectors[:, index]

print("Sorted Eigen Values:")
print(eigen_values.round(6))

print("Sorted Eigen Vectors:")
print(eigen_vectors.round(6))

explained_variance = eigen_values / eigen_values.sum()

print("Explained Variance Ratio:")
print(explained_variance.round(6))

print("Cumulative Explained Variance Ratio:")
print(np.cumsum(explained_variance).round(6))


principal_component = np.dot(standardized_data, eigen_vectors[:, :2])

pca_df = pd.DataFrame(principal_component, columns=['PC1', 'PC2'])

print("Principal Components:")
print(pca_df.round(6))


