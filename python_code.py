# Preprocess Data 
import pandas as pd 
df= pd.read_csv("heart_kaggle.csv")
df.head()
print("-------------------------Missing Values:-------------------------")
print(df.isna().sum())
print("\n-------------------------shape:-----------------------------------")
print(df.shape)
print("\n-------------------------Columns:------------------------------")
print(df.columns)
print("\n-------------------------Datatype:------------------------------")
print(df.dtypes)
print("\n-------------------------Taget values:-------------------------")
print(df['HeartDisease'].unique())
print("\n-----------------------Target Distribution:-------------------------")
print(df['HeartDisease'].value_counts())
print("\n--------------------Statistical Overview:-------------------------")
print(df.describe())
print("\n--------------------Corrupted Values: -------------------------")
print(df.apply(lambda x: pd.to_numeric(x, errors="coerce")).isnull().sum())

# Set target and features 
y=df['HeartDisease']
X=df.drop(columns='HeartDisease')

# Split into training and testing data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=28)

# Encode Categorical columns
cols = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
print('\nUnique Values :\n')
for col in cols:
    print(col, df[col].unique())

print("\nWe see no true natural ordering in the data ")

import category_encoders as ce 
encoder=ce.OneHotEncoder(use_cat_names=True)
#transform 
X_train_encoded=encoder.fit_transform(X_train)
X_test_encoded=encoder.transform(X_test)

# Scale numeric values 
from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()
num_cols = ['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']
X_train_encoded[num_cols]=scaler.fit_transform(X_train_encoded[num_cols])
X_test_encoded[num_cols] = scaler.transform(X_test_encoded[num_cols])

from sklearn.ensemble import RandomForestClassifier

# Create Model 
RF_model = RandomForestClassifier(n_estimators=200,random_state=42,oob_score=True,class_weight='balanced')
# we create 200 decision trees, oob= built in validation 

# Train Model
RF_model.fit(X_train_encoded,y_train)
# Make predictions 
RF_predictions = RF_model.predict(X_test_encoded)

# Performance evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

accuracy = accuracy_score(y_test, RF_predictions)
matrix = confusion_matrix(y_test, RF_predictions)
recall = recall_score(y_test, RF_predictions)

print("Accuracy:", accuracy)
print("Recall:", recall)
print(matrix)
print("TN FP FN TP =", matrix.ravel())

# Cross validation 
from sklearn.model_selection import cross_val_score

cv_accuracy = cross_val_score(RF_model, X_train_encoded, y_train, cv=10)
cv_recall = cross_val_score(RF_model, X_train_encoded, y_train, cv=10, scoring='recall')
cv_precision = cross_val_score(RF_model, X_train_encoded, y_train, cv=10, scoring='precision')
print("\n--- Cross Validation Results ---")
print("CV Accuracy Mean:", cv_accuracy.mean())
print("CV Recall Mean:", cv_recall.mean())
print("CV Precision Mean:", cv_precision.mean())

# Save 
import joblib

joblib.dump(RF_model, "heart_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
