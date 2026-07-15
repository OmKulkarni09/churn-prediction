import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,StratifiedKFold


DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # churn-prediction/

def load_data(url):
    """Load the raw Telco churn dataset."""
    df = pd.read_csv(url)
    return df


def clean_data(df):
    """Fix TotalCharges, drop customerID, encode target."""
    #Convert all the features to int and remove NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    #Remove non-contributing features
    df = df.drop(columns='customerID', errors='ignore')

    #Encode target variable
    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})

    return df


def prepare_features(df):
    """One-hot encode and split into X, y."""
    df_encoded = pd.get_dummies(df,drop_first=True)

    X=df_encoded.drop(columns='Churn', errors='ignore')
    y=df_encoded['Churn']

    return X,y



def evaluate(pipeline, X_test, y_test):
    """Print confusion matrix and classification report."""

    y_pred = pipeline.predict(X_test)

    print("------ CONFUSION MATRIX -----------\n")
    print(confusion_matrix(y_test,y_pred))

    print("\n ------- CLASSIFICATION REPORT ---------------------\n")
    print(classification_report(y_test,y_pred))

def build_pipeline():
    return Pipeline([
        ('scaler',StandardScaler()),
        ('model', LogisticRegression(max_iter=1000,class_weight='balanced'))
    ])

def cross_validate(pipeline,X,y):
    cv = StratifiedKFold(n_splits=5,shuffle = True, random_state = 42)
    recall_scores = cross_val_score(pipeline,X,y,cv=cv, scoring = 'recall')
    print(f"Mean recall: {recall_scores.mean():.3f} +/- {recall_scores.std():.3f}")



def main():
    df = clean_data(load_data(DATA_URL))
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,stratify=y
    )
    
    pipeline = build_pipeline()
    cross_validate(pipeline,X,y)
    pipeline.fit(X_train, y_train)
    evaluate(pipeline, X_test, y_test)


    final_model = build_pipeline()
    final_model.fit(X,y)
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "churn_pipeline.joblib"
    joblib.dump(final_model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()







