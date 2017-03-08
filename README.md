# tf-tutorials
some learned cases about using tensorflow

# Content
* Linear_regression
    * data

      randomly generate some points
    * model

      Y=WX+b
* Logistic_Regression

  * data

    download from https://www.kaggle.com/c/titanic/data.
    The Attribute like below:

        passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked
    Just choose the passenger_id, survived, pclass,to predict the sex.

  * model

    Y=1/(1+e^(WX+B))
