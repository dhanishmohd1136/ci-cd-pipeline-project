import pandas as pd 

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,f1_score


def train_model(data_path="data/data.csv"):

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Churn'])

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    num = X.select_dtypes(include=['number']).columns.tolist()
    cate = X.select_dtypes(include=['object']).columns.tolist()

    numerical_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='mean')),
        ('scaler',StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('Encoder',OneHotEncoder(handle_unknown='ignore'))
    ])


    preprocess = ColumnTransformer([
        ('num',numerical_pipeline,num),
        ('cate',categorical_pipeline,cate)
    ])

    model = Pipeline([
        ('preprocess',preprocess),
        ('classifier',LogisticRegression(max_iter=1000,random_state=43))
    ])

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=.2,
        random_state=43,
        stratify=y
    )

    model.fit(X_train,y_train)

    pred = model.predict(X_test)


    metrics = {
        "accuracy":accuracy_score(y_test,pred),
        "precision":precision_score(y_test,pred),
        "f1":f1_score(y_test,pred)
    }


    return model,X_train,metrics