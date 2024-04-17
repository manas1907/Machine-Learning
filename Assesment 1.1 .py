import pandas as pd
import matplotlib.pyplot as plt

# I. Load dataset
# Replace 'your_dataset.csv' with the actual path or URL of your dataset
dataset_path = 'data.csv'
df = pd.read_csv(dataset_path)

# II. Handle missing values
print("Missing values:")
print(df.isnull().sum())

# Replace missing numeric values with mean
df.fillna({'Pulse': df['Pulse'].mean(), 'Maxpulse': df['Maxpulse'].mean()}, inplace=True)

# Load the second dataset
dataset_path2 = 'data2.csv'
df2 = pd.read_csv(dataset_path2)

# III. Merge datasets using different types of joins

# Inner join
inner_join_df = pd.merge(df, df2, on='Duration', how='inner')

# Left join
left_join_df = pd.merge(df, df2, on='Calories', how='left')

# Right join
right_join_df = pd.merge(df, df2, on='Pulse', how='right')

# Outer join
outer_join_df = pd.merge(df, df2, on='Duration', how='outer')

# Analyze the impact of each type of join

# Print the shape of each merged dataframe to compare the number of rows and columns
print("Shape of Inner Join:", inner_join_df.shape)
print("Shape of Left Join:", left_join_df.shape)
print("Shape of Right Join:", right_join_df.shape)
print("Shape of Outer Join:", outer_join_df.shape)

# Display a sample of each merged dataframe
print("\nSample of Inner Join:")
print(inner_join_df.head())

print("\nSample of Left Join:")
print(left_join_df.head())

print("\nSample of Right Join:")
print(right_join_df.head())

print("\nSample of Outer Join:")
print(outer_join_df.head())


#. Create a new column and convert a categorical variable
# Create a new column by multiplying two existing columns
df['New_Calories'] = df['Duration'] * df['Calories']

# Convert a categorical variable into numerical representation (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['Date'])

# IV. Plotting
# Bar plot
df['Date'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Bar Plot of Date')
plt.xlabel('Date')
plt.ylabel('Count')
plt.show()

# Line plot
df['Pulse'].plot(kind='line', marker='o', linestyle='-', color='green')
plt.title('Line Plot of Pulse')
plt.xlabel('Index')
plt.ylabel('Pulse Values')
plt.show()

# Scatter plot
df.plot(kind='scatter', x='Duration', y='Calories', color='purple', alpha=0.5)
plt.title('Scatter Plot between Duration and Calories')
plt.xlabel('Duration')
plt.ylabel('Calories')
plt.show()

#. Visualize correlation matrix
correlation_matrix = df[['Duration', 'Pulse', 'Maxpulse', 'Calories']].corr()
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation='vertical')
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.title('Correlation Matrix of Numerical Columns')
plt.show()

# Histograms and box plots
# Histograms
df[['Duration', 'Calories']].hist(bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.suptitle('Histograms of Duration and Calories', x=0.5, y=1.02, ha='center', fontsize='large')
plt.show()

# Box plots
df[['Duration', 'Calories']].plot(kind='box', color=dict(boxes='darkgreen', whiskers='darkorange', medians='red', caps='gray'))
plt.title('Box Plot of Duration and Calories')
plt.show()

