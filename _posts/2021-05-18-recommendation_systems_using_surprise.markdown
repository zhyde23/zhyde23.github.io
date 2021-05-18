---
layout: post
title:      "Recommendation Systems using Surprise"
date:       2021-05-18 15:49:21 +0000
permalink:  recommendation_systems_using_surprise
---

There are many ways to go about creating a recommendation system but I will be solely explaining the context of the  Surprise model and how I utilized it within my project. This is more of a continuous, deeper exploration from my previous post [here](https://zhyde23.github.io/intro_to_recommendation_system).

To start, I imported the following:

```
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
%matplotlib inline

# !pip install requests (if you haven't already installed requests)

import requests
import nltk
# nltk.download('punkt')

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize

from surprise import Dataset, Reader
from surprise import BaselineOnly, SVD, SVDpp, NMF
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
from config import zach_key

# zach_key is my API key I saved in a text file within my notebook but had ignored when pushing onto my repo to avoid data leakage.
```

*Requests* helps you pull information from JSON files instead of saving them directly to your drive. This is very useful when you've finished your modeling and want to test it on a new data source for accuracy. (This will be shown later).

Now that all necessary libraries and tools are imported, we can start with the usual importing of data and begin our EDA process. Depending what you are trying to conclude or predict from your data will vary how you go about cleaning it and preparing your data to run through the surprise model. I'm going to fast-forward through this step due to EDA and cleaning being standardized with many educational tools and courses you will come across and get into the 'meat' of why you are here. **SURPRISE!!**

OK, so you have your dataframe created in a way that it will now pass through the surprise model. A friendly reminder, the model will only take in three columns at a time (e.g. ['user_id', 'buisness_id', 'stars'] ). This is really all you need for the model! It's a very user friendly model to utilize. The difficult part of it all, is the cleaning of data to get strong, relatable data to run through the model. With my project, I took a Yelp JSON file (two actually) and cleaned it in a way that I only had user_id's, the buisness_id that they reviewed, and the 'stars' score they gave of **Restaurants Only**. There were tons and tons of different businesses to pick from but restaurants category was the most rated business (37,139 counts) within the JSON files.

Here is where the modeling starts to happen:

```
# Defining the rating scale for the model

reader = Reader(rating_scale=(1.0, 5.0))

# Assigning the data I want to load into the model with the 'data' variable.

data = Dataset.load_from_df(reviews[['user_id', 'business_id', 'stars']], reader)
```

```
# Train-Test Split

train, test = train_test_split(data, test_size=.2)
```

Below, I created a function that would allow me to run the different types of models within surprise. There are more than these four; I just chose these at random. BaselineOnly() is a default choice to get, you guessed it, **a baseline** for all other models.

```
# Using the model selected to fit the trained data. 

def model_accuracy(model):
    model.fit(train)
# Assigning our model being used to test out the test set of data to 'predictions'
   
	 predictions = model.test(test)
# Outputing the Root Means Squared Error accuracy of the prediction
  
	return accuracy.rmse(predictions)
```

*The RMSE score will tell many stars the model is off from what it predicted to the actual stared review a user gave.*

```
Base = BaselineOnly()
model_accuracy(Base)
```
RMSE: 1.2711
```
svd = SVD()
model_accuracy(svd)
```
RMSE: 1.2752
```
NMF = NMF()
model_accuracy(NMF)
```
RMSE : 1.2839
```
SVDpp = SVDpp()
model_accuracy(SVDpp)
```
RMSE: 1.2839

**This is important!** You shouldn't use BaselineOnly() for final modeling even though it performed better than the others. It is strictly utilized as a reference for the rest of the models. I ended up using the next best, SVD().

I took a look at my data a little more and realized that a lot of the information I was giving the model wasn't very strong. The majority of the user's that had reviewed a buisness, reviewed less than 100 times. To make the data stronger for the modeling, I filtered out those that reviewed less than 100x which lowered the user count one million rows making it that much faster for the model to process the information on more useful user's that reviewed a restaurant.

I have a new dataset now that will be able to run more efficiently with my models. I need to re-run:

```
reader = Reader(rating_scale=(1.0, 5.0))
data_subset = Dataset.load_from_df(reviews_final[['user_id', 'business_id', 'stars']], reader)
```
**Notice my variable (reviews_final) changed from the previous (reviews)**
```
train, test = train_test_split(data_subset, test_size=.2)
```

I can now look at GridsearchCV to understand what parameters to utilize for my final modeling.

```
# These two types of parameters are the only relevant ones to utilize within this model.

svd_param_grid = {'n_factors': [5, 7, 10],
                  'reg_all': [0.002, 0.02, 0.1]}
```
```
# GridsearchCV will test out all possibilities of these parameter combinations.

svd_gs = GridSearchCV(algo_class = SVD, param_grid = svd_param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=2)
svd_gs.fit(data_subset)
```
```
# Outputting the results of the GridsearchCV.

pd.DataFrame(svd_gs.cv_results)
```
Gridsearch showed that the parameters (n_factors = 5, reg_all = 0.020) ran the best out of all combinations.

Next step is to re-run the models again using the subset data to compare to the originals and see how they compare.

Here were my RMSE results:

* BaselineOnly()  - 0.9815
* NMF - 1.0733
* SVD(w/o paramters) - 0.9828
* SVD(n_factors = 5, reg_all = 0.020) - 0.9804

(I chose not to re-run SVDpp() because it took over an hour to complete and knew I wasn't going to utilize it again.)

**The model's prediction is off by less than one star when comparing it to the actual review it was given!!!**

Now I told you that I would show how I used 'requests' within a JSON file that I hadn't saved directly to my drive so here is the function that created my predicted results for a user's top 5 restaurants they haven't reviewed yet as recommendations for them to explore.

```
def top_five_unrated_biz(user):

# This creates a list of buisnesses the user HAS reviewed and making it into a list.
    
		previous_ratings = reviews_final[reviews_final['user_id'] == user]['business_id'].tolist()

# Assigning a variable 'new_restaurants' to create a list of businesses the user hasn't reviewed yet.
    
		new_restaurants = list(filter(lambda x: x not in previous_ratings, restaurant_id))

    new_list = []
    for x in new_restaurants:

#This next line adds the predicted value to the businesses the user hasn't reviewed yet.
      
			new_list.append((svd.predict(user, x)[3], x))

# Sorting list of tuples based upon the first element of each tuple; getting the top 5 results. 
  
	top_five = sorted(new_list, key=lambda tup: tup[0], reverse = True)[:5]
    for y in top_five:
        rating = round(y[0], 2)
        
        
        url = 'https://api.yelp.com/v3/businesses/'+ y[1]

        headers = {
        'Authorization': 'Bearer {}'.format(zach_key),
            }

        response = requests.get(url, headers=headers).json()
        print(response['name'], rating)
```

Calling the function for user 'ltn9yaWIarK_o4DeMT1duA', I got the following:
```
top_five_unrated_biz('ltn9yaWIarK_o4DeMT1duA')
```
* Boston Roller Derby 4.74
* Boda Borg 4.74
* Mount Auburn Cemetery 4.72
* Jeni's Splendid Ice Creams 4.68
* Johnny's Pops 4.67

And there you have it!! A rec system accurate to 0.98 of a star when comparing the user's previous ratings of other restaurants they reviewed in the past. I hope this was insightful and helps you along your way to becoming a Data Scientist like I'm working towards. 

Thanks for the read and please feel free to reach out via[ LinkedIn ](https://www.linkedin.com/in/zachary-hyde-/)for questions or suggestions and check out the code [here](https://github.com/zhyde23/Capstone_Rec_Systems/blob/main/capstone-rec-system.ipynb) for this project.
