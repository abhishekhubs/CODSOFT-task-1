import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("Titanic-Dataset.csv")

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

drop_columns = ['Survived', 'Name', 'Ticket', 'Cabin']
X = df.drop(columns=drop_columns)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

print("Model               Accuracy")
print("------------------------------")
for name, acc in results.items():
    print(f"{name:<20} {acc:.4f}")

best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name}")

best_model = models[best_model_name]
best_preds = best_model.predict(X_test)
print(classification_report(y_test, best_preds))

cv_scores = cross_val_score(best_model, X, y, cv=5)
print("Mean Cross-Validation Accuracy (best model):", cv_scores.mean())
with open("model_comparison.txt", "w") as f:
    f.write("Model               Accuracy\n")
    f.write("------------------------------\n")
    for name, acc in results.items():
        f.write(f"{name:<20} {acc:.4f}\n")
    f.write(f"\nBest model: {best_model_name}\n")
    f.write(f"Mean cross-val accuracy (best model): {cv_scores.mean():.4f}\n")


gender_survival = df.groupby('Sex')['Survived'].value_counts().unstack()
gender_survival.plot(kind='bar')
plt.title("Survival Count by Gender")
plt.xlabel("Sex (0 = Male, 1 = Female)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("survival_by_gender.png")
plt.show()


pclass_survival_rate = df.groupby('Pclass')['Survived'].mean()
pclass_survival_rate.plot(kind='bar')
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("survival_rate_by_pclass.png")
plt.show()
