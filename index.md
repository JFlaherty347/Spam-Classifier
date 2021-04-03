# Spam-Classifier

A Naive Bayes Classifier that detects whether a text message is spam or not(ham). 

## Results

The classifier outputs the following results based on the unseen test data:

    Report: 
                precision    recall  f1-score   support

         ham       0.99      1.00      0.99      4333
        spam       1.00      0.93      0.96       680

    accuracy                           0.99      5013
    macro avg      0.99      0.96      0.98      5013
    weighted avg   0.99      0.99      0.99      5013

    Confusion Matrix: 
    [[4333    0]
    [  51  629]]

## Acknowledgements

This classifier is trained based on [data made available here through UCI's Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection#). Special thanks to them for making this data available!

