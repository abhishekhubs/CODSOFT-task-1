import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
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

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

cv_scores = cross_val_score(model, X, y, cv=5)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

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
