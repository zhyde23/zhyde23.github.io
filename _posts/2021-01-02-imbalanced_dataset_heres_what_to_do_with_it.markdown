---
layout: post
title:      "Imbalanced Dataset? Here's what to do with it."
date:       2021-01-02 05:55:43 +0000
permalink:  imbalanced_dataset_heres_what_to_do_with_it
---


There will be numerous times you will come across an imbalanced dataset. If you were to run accuracy and recall scores of your model, you will likely see that it's very high. Think of this: let's say you are building a predictive model to accurately measure whether or not your customer will continue services with your company based on x-y-z factors. You have a dataset that has an imbalanced count of those who continue services and those who terminate them. 90% of your customers continue their membership while 10% cancel. When you run your accuracy score's, you will most likely get a high number (0.90) because the computer is predicting 9/10 times the customer continues services. That's awesome! You think, "Wow! This is a super accurate model". Then you continue to check on the recall score of that same model. Turns out your score is 0.45. Oh no! When it comes to new information entering the model, the computer hasn't learned to look for certain features. An example of features that you might utilize are age of account, how often the account's been utilized on a monthly bases, or the number of customer service calls it's had during its lifespan. These are just to name a few to get the idea of customer retention features you might see within your next dataset. Your model won't be able to accurately predict if the age of an account is high, the customer will more likely continue services. Maybe if the customer had more than five service calls, they would have a higher chance of canceling the subscription. That's why it's so important to balance the weight of the yes's and no's for your target "customer churn". When the weight distribution is the same, the model you end up choosing can fairly assess both end results of the customer and learn the features of what goes into a customer staying or leaving.

There are two ways to make this necessary adjustment to your trainning set.


(IMPORTANT: DO NOT APPLY THESE TO YOUR TEST SET)


**SMOTE** and **NEARMISS** are methods utilized with imbalanced datasets. 

SMOTE, or Synthetic Minority Oversampling TEchnique, increases your smaller target data points by "synthetically" mapping out more of it by following linear patterns of the current points, increasing your minority count to that of your majority. Below is the code you'll want to utilize:

```
# You might need to install imblearn if you haven't already

pip install imblearn

from imblearn.over_sampling import SMOTE

# Before fitting SMOTE, check the y_train values. This gives you a starting point to base your adjustment off of.

y_train.value_counts()


# Let us fit SMOTE: (You can check out all the parameters from here)

smt = SMOTE()
X_train, y_train = smt.fit_sample(X_train, y_train)

# Now check the amount of records in each category:
np.bincount(y_train)


# Now fitting the classifier and testing for an accuracy score.
accuracy_score(y_test, y_pred)

# Check the confusion matrix jsut to verify the change:
confusion_matrix(y_test, y_pred)

# Now check your recall.
recall_score(y_test, y_pred)

```

NearMiss is the other technique for adjusting an imbalanced dataset by underfitting your majority target value. The approach here is reducing the amount of values the model will observe within the majority group to the same amount as the minority, keeping the two balanced.

```
from imblearn.under_sampling import NearMiss

# Fit NearMiss:

nr = NearMiss()

X_train, y_train = nr.fit_sample(X_train, y_train)

# Check the amount of records in each category.

np.bincount(y_train)

# The majority class has been reduced to the total number of the minority class. This now creates an even balance between the two.

# Now fit the classifier and test the model.

confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

recall_score(y_test, y_pred)
```

I hope this read has given you some insight on the importance of correcting imbalanced datasets and how to make those corrections.

If you'd like to read further in depth on this technique, here's the original research paper titled "[SMOTE: Synthetic Minority Over-sampling Technique](https://jair.org/index.php/jair/article/view/10302)"

