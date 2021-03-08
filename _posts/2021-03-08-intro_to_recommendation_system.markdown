---
layout: post
title:      "Intro to Recommendation System"
date:       2021-03-08 05:52:12 +0000
permalink:  intro_to_recommendation_system
---


What is a recommendation system? It’s pretty straight forward thinking. Think of what you use every day. Products like Amazon/Prime, Netflix, Hulu or whenever a website discloses that they use cookies when you browse their site; all these types of systems utilize machine learning and different algorithms to effectively market to you specifically from the device you work off of. What these software companies are doing is learning the consumer habits and frequently viewed topics of interest to better cater to those interests, effectively marketing or advertising product similar to what has been previously viewed or purchased. Amazon is one of the most efficient at doing so when you quickly think about it. I’m always looking for that ‘thing’ on their app or website. I look at what’s similar to item ‘x’ and what others have purchased with this item or have viewed because they viewed this item. It’s crazy to think about how software has evolved into almost becoming mind readers! I hear jokes about “I was thinking of buying this speaker the other day and when I went onto Amazon, there it was! I hadn’t even searched for anything yet and it suggested the speaker as an item I might like”. 

The way recommendation systems work is utilizing already known information about the consumer(s) and building a model that is able to train and test from it to accurately predict something else of similar interest. I found it so fascinating when I started to learn about it within my Data Science bootcamp. Recommendation systems are all around us within the internet and the systems that run/stream off of it. I will share with you a very useful step-by-step tutorial I saw as part of my lessons through bootcamp utilizing the surprise library which helped me through my first recommendation project.


Fitting and Predicting with Surprise
* Install surprise if you haven't, and import the usual libraries.

```
# !pip install surprise
# import libraries

import numpy as np
import pandas as pd

from surprise import Dataset, Reader
from surprise import SVD```

[Check out other models to use within suprise](https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html)

```
from surprise import accuracy

from surprise.model_selection import cross_validate, train_test_split
```

* Load in the dataset
Surprise has the dataset built in. You might need to download the dataset so follow the instructions in the code output! Unfortunately, the Surprise data format doesn't let us inspect the data, but here is the [documentation](https://grouplens.org/datasets/movielens/100k/).

```
data = Dataset.load_builtin('ml-100k')

# train-test split
train, test = train_test_split(data, test_size=.2)

train
```

* Run the default Singular Value Decomposition Model!

```
svd = SVD()
svd.fit(train)
predictions = svd.test(test)

accuracy.rmse(predictions)
```

* Make a prediction!

```
uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
iid = str(302)

# get a prediction for specific users and items.
pred = svd.predict(uid, iid, r_ui=4, verbose=True)
```

Applying Surprise
* How does Surprise [take in](https://surprise.readthedocs.io/en/stable/getting_started.html#use-a-custom-dataset) your data?
The dataset we'll use is a subset of the Yelp [Open Dataset](https://www.yelp.com/dataset) that's already been joined and cleaned.

```
yelp = pd.read_csv('yelp_reviews.csv').drop(['Unnamed: 0'], axis = 1)

yelp.head()
```

* Inspecting the dataset:
Here's where you'd do a comprehensive EDA!

```
print('Number of Users: ', len(yelp['user_id'].unique()))
print('Number of Businesses: ', len(yelp['business_id'].unique()))
```

1. What's the distribution of ratings? i.e. How many 1-star, 2-star, 3-star reviews?
2. How many reviews does a restaurant have?
3. How many reviews does a user make?

```
yelp['stars'].value_counts()

yelp['business_id'].value_counts()
```


* Reading in the dataset and prepping data

```
# Instantiate a 'Reader' to read in the data so Surprise can use it
reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(yelp[['user_id', 'business_id', 'stars']], reader)

trainset, testset = train_test_split(data, test_size=.2)

trainset
```

* Fitting and evaluating models
Here, let's assume that we've tuned all these hyperparameters using GridSearch, and we've arrived at our final model.

```
final = SVD(n_epochs=20, n_factors=1, biased=True, 
              lr_all=0.005, reg_all=0.06)

final.fit(trainset)

predictions = final.test(testset)

predictions[:3]

accuracy.rmse(predictions)
accuracy.mae(predictions)
```

* Making Predictions (again)
Unfortunately, this dataset has a convoluted string as the user/business IDs.

```
yelp['user_id'][55]

yelp['business_id'][123]

final.predict(yelp['user_id'][55], yelp['business_id'][13])
```

* What else?
Surprise has sample code where you can get the top n recommended [items](https://surprise.readthedocs.io/en/stable/FAQ.html) for a user. 

Resources:

* The structure of our lesson on recommendation engines is based on Chapter 9 of Mining of Massive Datasets: http://infolab.stanford.edu/~ullman/mmds/book.pdf
* Libraries for coding recommendation engines:
    * Surprise: https://surprise.readthedocs.io/en/stable/index.html


In summary, the set up of surprise is very similar to many other libraries utilized as a Data Scientist. You will always need to import or web scrape to start with some type of information, clean it up by removing null values and unuseful information or columns. Once that is complete, start to make a few graphs or dataframe tables to get visuals of your data and make educated hypotheses of what can happen to your information as you start endering it into your models and manipulating it. Always check for accuracy when running your dataframe through each model to establish the one you will utilize for your final edits and take advantage of the GridSearchCV to find your best parameters to the final model. Once you have fulfilled these steps, you have an accurate recommendation system based on the initial raw data that it was provided with in the first place.

