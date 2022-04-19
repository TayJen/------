from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from time import time
import pandas as pd


def train_model(data, model):
    X_train, X_test, y_train, y_test = train_test_split(data.drop('is_fake', axis=1), data['is_fake'],
                                                        test_size=0.2, random_state=13)

    t0 = time()
    model.fit(X_train, y_train)
    t1 = time()
    y_pred = model.predict(X_test)
    t2 = time()

    time_train = t1 - t0
    time_pred = t2 - t1

    print(model)
    print("Training time: %fs; Prediction time: %fs \n" % (time_train, time_pred))
    print('Accuracy score train set :', model.score(X_train, y_train))
    print('Accuracy score test set  :', accuracy_score(y_test, y_pred),'\n')
    print(classification_report(y_test, y_pred))
    
    print('\n -------------------------------------------------------------------------------------- \n')


def main():
    path = 'new data\\new_df_train.csv'
    df = pd.read_csv(path)

    for model in [LogisticRegression(), KNeighborsClassifier(),
                  RandomForestClassifier(), XGBClassifier()]:
        train_model(df, model)


main()