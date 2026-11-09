import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

def train():
    
    df = pd.read_csv("data/customer_churn_dataset-testing-master.csv")
    x = df.drop('Churn',axis=1)
    y =  df['Churn']

    num = x.select_dtypes(include=['int64','float64']).columns.tolist()
    cat = x.select_dtypes(include=['str','object']).columns.tolist()

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    num_pipe = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    cat_pipe = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('one',OneHotEncoder(handle_unknown='ignore',drop='first'))])
    preprocess = ColumnTransformer([('num',num_pipe,num),('cat',cat_pipe,cat)])
    model = Pipeline([('preprocess',preprocess),("model",XGBClassifier())])
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    recall = recall_score(y_test,pred)
    prec = precision_score(y_test,pred)
    print(acc,f1,recall,prec)

if __name__ == "__main__":
    train()