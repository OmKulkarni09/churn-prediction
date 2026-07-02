import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

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

def train_model(X_train, y_train):
    """Scale features and train class-balanced logistic regression."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000,class_weight='balanced')
    model.fit(X_train,y_train)

    return model,scaler



def evaluate(model, scaler, X_test, y_test):
    """Print confusion matrix and classification report."""
    X_test = scaler.transform(X_test)

    y_pred = model.predict(X_test)

    print("------ CONFUSION MATRIX -----------\n")
    print(confusion_matrix(y_test,y_pred))

    print("\n ------- CLASSIFICATION REPORT ---------------------\n")
    print(classification_report(y_test,y_pred))


def main():
    df = load_data(DATA_URL)
    df = clean_data(df)

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model, scaler = train_model(X_train, y_train)
    evaluate(model, scaler, X_test, y_test)

if __name__ == "__main__":
    main()







