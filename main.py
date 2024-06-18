import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


dataset = pd.read_csv('../healthcare-dataset-stroke-data.csv', sep=',', encoding='cp1252')

dataset.dropna(inplace=True)

dataset.rename(columns={'Residence_type': 'residence_type'}, inplace=True)

dataset['age'] = pd.to_numeric(dataset['age'], errors='coerce')
dataset['bmi'] = pd.to_numeric(dataset['bmi'], errors='coerce')
dataset['gender'] = dataset['gender'].map({'Male': 1, 'Female': 0})
dataset['ever_married'] = dataset['ever_married'].map({'Yes': 1, 'No': 0})
dataset['residence_type'] = dataset['residence_type'].map({'Urban': 1, 'Rural': 0})
dataset['smoking_status'] = dataset['smoking_status'].map({'smokes': 1, 'formerly smoked': 1, 'Unknown': 0, 'never smoked': 0})

total_dataset_rows = dataset.shape[0]
print("Total Rows:", total_dataset_rows)

gender_counts = dataset['gender'].value_counts()

dataset['stroke_risk'] = 0


dataset['stroke_risk'] += (dataset['age'] > 60)
dataset['stroke_risk'] += dataset['hypertension']
dataset['stroke_risk'] += dataset['heart_disease']
dataset['stroke_risk'] += (dataset['avg_glucose_level'] >= 150)
dataset['stroke_risk'] += (dataset['bmi'] >= 30)
dataset['stroke_risk'] += (dataset['smoking_status'] == 1)
dataset['stroke_risk'] += (dataset['gender'] == 0)
dataset['stroke_risk'] += (dataset['ever_married'] == 0)
dataset['stroke_risk'] += (dataset['residence_type'] == 1)

dataset['stroke'] = (dataset['stroke_risk'] >= 5).astype(int)

total_strokes = dataset['stroke'].sum()
print("Total Strokes:", total_strokes)
print("Percentage Strokes:", ((total_strokes / total_dataset_rows) * 100).round(2), "%")

plt.figure(figsize=(12, 8))
sns.histplot(dataset['age'], bins=30, kde=True, color='blue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=dataset, x='hypertension', hue='stroke')
plt.title('Stroke by Hypertension Status')
plt.xlabel('Hypertension (0: No, 1: Yes)')
plt.ylabel('Count')
plt.legend(title='Stroke', labels=['No', 'Yes'])
plt.show()

numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = dataset[numeric_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


X = dataset.drop(columns=['id', 'stroke', 'stroke_risk', 'work_type'])
y = dataset['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)

dt_gini.fit(X_train, y_train)
dt_entropy.fit(X_train, y_train)

y_pred_gini = dt_gini.predict(X_test)
y_pred_entropy = dt_entropy.predict(X_test)

accuracy_gini = accuracy_score(y_test, y_pred_gini)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)

print("Gini Accuracy:", accuracy_gini)
print("Entropy Accuracy:", accuracy_entropy)

print("\nClassification Report for Gini:")
print(classification_report(y_test, y_pred_gini))
print("\nClassification Report for Entropy:")
print(classification_report(y_test, y_pred_entropy))

print("\nConfusion Matrix for Gini:")
print(confusion_matrix(y_test, y_pred_gini))
print("\nConfusion Matrix for Entropy:")
print(confusion_matrix(y_test, y_pred_entropy))

features = X.columns

importances_gini = dt_gini.feature_importances_
indices_gini = importances_gini.argsort()[::-1]
print("\nFeature importances using Gini:")
for f in range(X.shape[1]):
    print(f"{features[indices_gini[f]]}: {importances_gini[indices_gini[f]]}")

importances_entropy = dt_entropy.feature_importances_
indices_entropy = importances_entropy.argsort()[::-1]
print("\nFeature importances using Entropy:")
for f in range(X.shape[1]):
    print(f"{features[indices_entropy[f]]}: {importances_entropy[indices_entropy[f]]}")

plt.figure(figsize=(16, 6), dpi=670)
plot_tree(dt_gini, filled=True, feature_names=features, class_names=['No Stroke', 'Stroke'], rounded=True)
plt.title('Decision Tree using Gini')
plt.show()

plt.figure(figsize=(16, 8), dpi=500)
plot_tree(dt_entropy, filled=True, feature_names=features, class_names=['No Stroke', 'Stroke'], rounded=True)
plt.title('Decision Tree using Entropy')
plt.show()
