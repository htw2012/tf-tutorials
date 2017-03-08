# tf-tutorials
some learned cases about using tensorflow

# Content
* Linear_regression
    * data

      randomly generate some points
    * model

      Y=WX+b,where y is a real value
    * API

      tf.mul(X, W) + b
* Logistic_Regression

  * data

    download from https://www.kaggle.com/c/titanic/data.
    The Attribute like below:

        passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked
    Just choose the passenger_id, survived, pclass,to predict the sex.

  * model

    Y=1/(1+e^(WX+b)),where y is 0 or 1.
  * API

    tf.nn.sigmoid_cross_entropy_with_logits(tf.matmul(x_in, W) + b, y_in)

* Softmax_Classification

  * data

    iris data,it's UCI data,also you can download from https://www.kaggle.com/uciml/iris

  * model

    Y=1/(1+e^(WX+b)),where  you can use multiple labels
  * API

    tf.nn.sparse_softmax_cross_entropy_with_logits
    tf.nn.softmax_cross_entropy_with_logit
