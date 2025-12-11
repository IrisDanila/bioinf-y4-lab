# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("/workspaces/bioinf-y4-lab/data/sample/tissue_gene_expression_demo.csv")  # Create a subset from the GTEx Gene expression dataset or use dataset from previous lab example
print("Dataset Preview:")
print(data.head())

# Additional dataset information
print(f"\nDataset shape: {data.shape}")
print(f"\nColumn names: {list(data.columns)}")
print(f"\nTissue type distribution:\n{data['Tissue_Type'].value_counts()}")
print(f"\nDataset info:")
print(data.info())

# Split data into features (X) and target (y)
# Drop non-numeric identifiers to avoid casting errors during model fitting
X = data.drop(columns=['sample_id', 'Tissue_Type'])
y = data['Tissue_Type']

# Encode target labels if necessary (e. g., string to numeric)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_labels = list(range(len(label_encoder.classes_)))
print("Classes:", label_encoder.classes_)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test. shape[0]} samples")
print(f"Number of features: {X_train.shape[1]}")

# Initialize and train a Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("\n" + "="*80)
print("INITIAL MODEL EVALUATION")
print("="*80)
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=class_labels,
    target_names=label_encoder.classes_,
    zero_division=0,
))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Tissue Type Prediction")
plt.savefig("confusion_matrix_initial.png", dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance
feature_importances = rf_model.feature_importances_
plt.figure(figsize=(12, 8))
# Sort features by importance for better visualization
feature_importance_sorted = pd.DataFrame({
    'Feature':  X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=True)
plt.barh(feature_importance_sorted['Feature']. tail(20), 
         feature_importance_sorted['Importance'].tail(20))
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig("feature_importance_all.png", dpi=300, bbox_inches='tight')
plt.show()

#output classification report, confusion matrix and feature importance plot
#Use GridSearchCV to optimize parameters like n_estimators and max_depth
#Experiment with selecting the top 10 most important features and retraining the model

# ===== ADDITIONAL IMPLEMENTATION STARTS HERE =====

print("\n" + "="*80)
print("HYPERPARAMETER OPTIMIZATION WITH GRIDSEARCHCV")
print("="*80)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

print("\nParameter grid for GridSearchCV:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

# Initialize GridSearchCV
cv_splits = max(2, min(3, len(y_train)))  # Cap splits to available samples
cv_strategy = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
print(f"\nPerforming GridSearchCV with {cv_splits}-fold CV (this may take a few minutes)...")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv_strategy,
    n_jobs=-1,  # Use all available cores
    verbose=2,
    scoring='accuracy'
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and score
print("\n" + "-"*80)
print("GridSearchCV Results:")
print("-"*80)
print(f"Best parameters:  {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train the best model
best_rf_model = grid_search.best_estimator_

# Predict with optimized model
y_pred_optimized = best_rf_model. predict(X_test)

# Evaluate optimized model
print("\nOptimized Model - Classification Report:")
print(classification_report(
    y_test,
    y_pred_optimized,
    labels=class_labels,
    target_names=label_encoder.classes_,
    zero_division=0,
))

# Confusion Matrix for optimized model
conf_matrix_optimized = confusion_matrix(y_test, y_pred_optimized, labels=class_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_optimized, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Optimized Random Forest")
plt.savefig("confusion_matrix_optimized.png", dpi=300, bbox_inches='tight')
plt.show()

# Compare accuracies
initial_accuracy = accuracy_score(y_test, y_pred)
optimized_accuracy = accuracy_score(y_test, y_pred_optimized)

print(f"\nModel Performance Comparison:")
print(f"  Initial Model Accuracy:     {initial_accuracy:.4f}")
print(f"  Optimized Model Accuracy:  {optimized_accuracy:.4f}")
print(f"  Improvement:               {(optimized_accuracy - initial_accuracy):.4f}")

# ===== FEATURE SELECTION:  TOP 10 FEATURES =====

print("\n" + "="*80)
print("FEATURE SELECTION - TOP 10 MOST IMPORTANT FEATURES")
print("="*80)

# Get feature importances from optimized model
feature_importances_optimized = best_rf_model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances_optimized
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10).to_string())

# Select top 10 features
top_10_features = feature_importance_df.head(10)['Feature'].values
print(f"\nSelected features: {list(top_10_features)}")

# Visualize top 10 features
plt.figure(figsize=(10, 6))
top_10_data = feature_importance_df.head(10)
plt.barh(top_10_data['Feature']. values[: :-1], top_10_data['Importance'].values[::-1])
plt.xlabel("Feature Importance")
plt.ylabel("Gene")
plt.title("Top 10 Most Important Features")
plt.tight_layout()
plt.savefig("feature_importance_top10.png", dpi=300, bbox_inches='tight')
plt.show()

# Create new dataset with only top 10 features
X_train_top10 = X_train[top_10_features]
X_test_top10 = X_test[top_10_features]

print(f"\nReduced dataset shape:")
print(f"  Training set:  {X_train_top10.shape}")
print(f"  Test set:      {X_test_top10.shape}")

# ===== RETRAIN MODEL WITH TOP 10 FEATURES =====

print("\n" + "-"*80)
print("Retraining Random Forest with Top 10 Features")
print("-"*80)

# Train a new model with top 10 features using best parameters
rf_model_top10 = RandomForestClassifier(
    **grid_search.best_params_,
    random_state=42
)

rf_model_top10.fit(X_train_top10, y_train)

# Predict with reduced feature model
y_pred_top10 = rf_model_top10.predict(X_test_top10)

# Evaluate the reduced feature model
print("\nTop 10 Features Model - Classification Report:")
print(classification_report(
    y_test,
    y_pred_top10,
    labels=class_labels,
    target_names=label_encoder.classes_,
    zero_division=0,
))

# Confusion Matrix for top 10 features model
conf_matrix_top10 = confusion_matrix(y_test, y_pred_top10, labels=class_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_top10, annot=True, fmt='d', cmap='Purples',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Top 10 Features Model")
plt.savefig("confusion_matrix_top10.png", dpi=300, bbox_inches='tight')
plt.show()

# Feature importance for top 10 model
feature_importances_top10 = rf_model_top10.feature_importances_
plt.figure(figsize=(10, 6))
top10_reordered = pd.DataFrame({
    'Feature': top_10_features,
    'Importance': feature_importances_top10
}).sort_values(by='Importance', ascending=True)
plt.barh(top10_reordered['Feature'], top10_reordered['Importance'])
plt.xlabel("Feature Importance (Normalized)")
plt.ylabel("Gene")
plt.title("Feature Importance in Top 10 Features Model")
plt.tight_layout()
plt.savefig("feature_importance_top10_retrained.png", dpi=300, bbox_inches='tight')
plt.show()

# ===== FINAL COMPARISON =====

print("\n" + "="*80)
print("FINAL MODEL COMPARISON SUMMARY")
print("="*80)

top10_accuracy = accuracy_score(y_test, y_pred_top10)

comparison_data = {
    'Model': ['Initial (100 trees, all features)', 
              'Optimized (GridSearchCV, all features)', 
              'Top 10 Features (Optimized params)'],
    'Number of Features': [X. shape[1], X.shape[1], 10],
    'Accuracy':  [initial_accuracy, optimized_accuracy, top10_accuracy]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))

# Visualize comparison
plt.figure(figsize=(12, 6))

# Subplot 1: Accuracy comparison
plt.subplot(1, 2, 1)
colors = ['#3498db', '#2ecc71', '#9b59b6']
plt.bar(range(3), comparison_df['Accuracy'], color=colors, alpha=0.7, edgecolor='black')
plt. xticks(range(3), ['Initial', 'Optimized', 'Top 10\nFeatures'], rotation=0)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim([min(comparison_df['Accuracy']) - 0.1, 1.0])
for i, v in enumerate(comparison_df['Accuracy']):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Subplot 2: Feature count vs accuracy
plt.subplot(1, 2, 2)
plt.scatter(comparison_df['Number of Features'], comparison_df['Accuracy'], 
           s=200, c=colors, alpha=0.7, edgecolors='black', linewidths=2)
for i, model in enumerate(comparison_df['Model']):
    plt.annotate(model. split('(')[0].strip(), 
                (comparison_df['Number of Features'][i], comparison_df['Accuracy'][i]),
                xytext=(10, 10), textcoords='offset points', fontsize=9)
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Feature Count vs Model Accuracy')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nKey Findings:")
print(f"1. GridSearchCV improved accuracy by:  {(optimized_accuracy - initial_accuracy)*100:.2f}%")
print(f"2. Using only top 10 features ({(10/X. shape[1])*100:.1f}% of features):")
print(f"   - Accuracy: {top10_accuracy:.4f}")
print(f"   - Change from optimized model: {(top10_accuracy - optimized_accuracy)*100:.2f}%")
if top10_accuracy >= optimized_accuracy - 0.05:
    print(f"3. Feature reduction benefit:  Simpler model with minimal accuracy loss")
    print(f"   - Model complexity reduced by {((X. shape[1] - 10)/X.shape[1])*100:.1f}%")
else:
    print(f"3. Feature reduction trade-off: Consider using more features for better accuracy")

print("\nAll visualizations saved:")
print("  - confusion_matrix_initial.png")
print("  - confusion_matrix_optimized.png")
print("  - confusion_matrix_top10.png")
print("  - feature_importance_all.png")
print("  - feature_importance_top10.png")
print("  - feature_importance_top10_retrained.png")
print("  - model_comparison.png")

# Save the comparison results to a CSV file
comparison_df.to_csv("model_comparison_results.csv", index=False)
print("  - model_comparison_results.csv")

# Save top 10 features to a file
top_10_df = feature_importance_df.head(10)
top_10_df.to_csv("top_10_features.csv", index=False)
print("  - top_10_features.csv")

print("\n" + "="*80)