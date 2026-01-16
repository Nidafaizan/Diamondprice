import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('diamonds.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Create target variable - classify diamond prices into categories
df['price_category'] = pd.cut(df['price'], bins=3, labels=[
                              'Low', 'Medium', 'High'])

# Encode categorical variables
le_cut = LabelEncoder()
le_color = LabelEncoder()
le_clarity = LabelEncoder()
le_price_category = LabelEncoder()

df['cut_encoded'] = le_cut.fit_transform(df['cut'])
df['color_encoded'] = le_color.fit_transform(df['color'])
df['clarity_encoded'] = le_clarity.fit_transform(df['clarity'])
df['price_category_encoded'] = le_price_category.fit_transform(
    df['price_category'])

# Select features and target
feature_columns = ['carat', 'cut_encoded', 'color_encoded',
                   'clarity_encoded', 'depth', 'table', 'x', 'y', 'z']
X = df[feature_columns]
y = df['price_category_encoded']

print("\nFeatures shape:", X.shape)
print("Target distribution:")
print(df['price_category'].value_counts())

# Apply MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Dictionary of models to test
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=100, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', C=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
}

# Train and evaluate models
results = {}

print("\n" + "="*80)
print("MODEL EVALUATION RESULTS")
print("="*80)

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 50)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred
    }

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']

print("\n" + "="*80)
print(f"BEST MODEL: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")
print("="*80)

# Detailed report for best model
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['y_pred']

print("\nDetailed Classification Report (Best Model):")
print("-" * 50)
print(classification_report(y_test, y_pred_best,
      target_names=['Low', 'Medium', 'High']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# Feature importance for tree-based models
if hasattr(best_model, 'feature_importances_'):
    print("\nFeature Importance (Top 5):")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head())

# Save the best model
with open('best_classification_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✓ Best model saved as 'best_classification_model.pkl'")
print("✓ Scaler saved as 'scaler.pkl'")
