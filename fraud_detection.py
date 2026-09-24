import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics import ConfusionMatrixDisplay


# -----------------------------
# 1. Load the dataset
# -----------------------------

data = pd.read_csv("data/creditcard.csv")

print("Dataset shape:", data.shape)
print("\nClass distribution:")
print(data["Class"].value_counts())

class_counts = data["Class"].value_counts()

plt.bar(["Legitimate", "Fraud"], class_counts.values)
plt.title("Distribution of Credit Card Transactions")
plt.ylabel("Number of Transactions")
plt.savefig("images/class_distribution.png", bbox_inches="tight")
plt.close()


# -----------------------------
# 2. Separate features and target
# -----------------------------

X = data.drop("Class", axis=1)
y = data["Class"]


# -----------------------------
# 3. Split into training and test sets
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# -----------------------------
# 4. Scale data for Logistic Regression
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# 5. Logistic Regression
# -----------------------------

print("\n--- Logistic Regression ---")

logistic_model = LogisticRegression(
    solver="liblinear",
    max_iter=1000
)

logistic_model.fit(X_train_scaled, y_train)

logistic_predictions = logistic_model.predict(X_test_scaled)

logistic_precision = precision_score(y_test, logistic_predictions)
logistic_recall = recall_score(y_test, logistic_predictions)
logistic_f1 = f1_score(y_test, logistic_predictions)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))

print("\nClassification Report:")
print(classification_report(y_test, logistic_predictions))


# -----------------------------
# 6. Random Forest
# -----------------------------

print("\n--- Random Forest ---")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

metrics = ["Precision", "Recall", "F1 Score"]

logistic_scores = [
    logistic_precision,
    logistic_recall,
    logistic_f1
]

rf_scores = [
    rf_precision,
    rf_recall,
    rf_f1
]

x = range(len(metrics))

plt.figure()

plt.bar(
    [i - 0.2 for i in x],
    logistic_scores,
    width=0.4,
    label="Logistic Regression"
)

plt.bar(
    [i + 0.2 for i in x],
    rf_scores,
    width=0.4,
    label="Random Forest"
)

plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Fraud Detection Model Comparison")
plt.legend()

plt.savefig("images/model_comparison.png", bbox_inches="tight")
plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_predictions,
    display_labels=["Legitimate", "Fraud"]
)

plt.title("Random Forest Confusion Matrix")
plt.savefig("images/confusion_matrix.png", bbox_inches="tight")
plt.close()
