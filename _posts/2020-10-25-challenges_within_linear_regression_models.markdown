---
layout: post
title:      "Challenges within Linear Regression Models"
date:       2020-10-26 02:47:48 +0000
permalink:  challenges_within_linear_regression_models
---

![](https://miro.medium.com/max/1376/1*G1Y_-X14q2xMVHlUuaUUdA.png)

For those of you that are beginning your first projects on Linear Regression, you may have more questions than answeres like I did at this point. As I complete my first regression model, I want to walk you through what I found helpful and what I struggled with and how I was able to wrap my head around these blockers.

You'll here time and time again that the internet search is your best friend. It is 100% accurate because the tech industry is constantly evolving/updating. I believe I mentioned this in one of my later posts; data science is unique with the mentality that everyone within the industry is out to help one another and not necessarily compete against each other. What I mean is, as you evolve from beginner to tenured analyst/scientist, you will have many questions about how to type out certain code or run into a situation that you have no idea how to solve. The internet is the most utilized tool at your disposal to most of your questions! Searching for a solution to your problem 9/10 times will lead you to your solution. The mentality to this? Data Science is 'as strong as the weakest link'. The more we know and collaborate together, the stronger and more accurate we can all get as the data world progresses and reaches higher milestones.

But enough about that. Here's what you all came here for; to understand how to go about starting your project!

First off, you'll want to import all necessary tools before you start running your formulas and manipulating your data set. As a headstart, I've included most of my utilized tools with some graphing for this project below.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline
plt.style.use('ggplot')

import scipy.stats as stats

import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
```

Your next steps would be to begin your EDA/cleaning of the dataframe you were provided or scrapped off the web. The most useful methods I used frequently were:

* `.info() ` -provides you with a list of your DataFrame columns showing what type each represents (i.e. object, float, integer), how many null values you're working against,  and the size of the DataFrame.  
* `.describe()` -returns a nicely viewed DataFrame similar to an exel file. What you'll be most interested at observing are: standard deviation, minimum, maximum, mean and quartile representations of each column. 
* `.fillna()` -getting rid of those pesky NaN values that doesn't provide you with 'clean' data. There are a few examples you'll want to include within (): 0 if it's truely representing no value or the mean **of your selected column** to have the best data representation when there is missing information.
* `.value_counts()` -simply provides you with a list of unique value counts within you're selected code. Example: df['column_name'].value_counts()

After your EDA, you'll want to start plotting out the data to have an understanding how it looks linearly on a graph. `plt.scatter()` will simply give you whatever your X, Y values you assigned to a scatter plot graph. This allows you to observe what type of coorelation the data has (possitive, negative or neutral). Then you can start to code out your best fit (regression) line to this graph. I ran into some of my blockers here. After scowering through course notes and lab work previously assigned, I was able to successfully complete this task using code below:

```
def calc_slope(xs, ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) / ((np.mean(xs)**2) - np.mean(xs*xs)))
    
    return m
calc_slope(X, y)
```
```
def best_fit(xs, ys):
    
    m = calc_slope(xs, ys)
    c = np.mean(ys) - m*np.mean(xs)
    
    return m, c

m, c = best_fit(X, y)
m, c
```
```
def reg_line (m, c, xs):
    
    return [(m*x)+c for x in xs]
regression_line = reg_line(m, c, X)
```
```
plt.scatter(X, y, color='#003F72', label= 'Data points')
plt.plot(X, regression_line, label= 'Regression Line')
plt.legend()
```

You have a better understanding what your data represents numerically and visually now. The next steps I won't go into detail so much as above only to cue your searches and utilize some of my linked sources below.

Essentially, these next steps that follow will be to create dummy variables to your categorical columns. This method, `.get_dummies()`, allows you to separate your categories into numerical values (changing how the computer sees them as objects into integers). Without doing so, will cause your data to be misinterpereted.

Last steps needed:
* OLS 
* Log Transformation
* Creating a hypothesis for your test
* Train-test split
* Q-Q plots
* Regression Model Validation
* Cross Validation

This concludes my initial set up of Linear Regression Modeling and some of the trouble points I came into contact with along the way. 

Thank you for taking the time to read and happy learning!


**I've included a few sources that I utilized for my project below.**

*Towards Data Science* has been the most helpful along with some YouTube videos with walk throughs on linear regression concept and the statistics behind it. [Towards Data Science](https://open.spotify.com/show/63diy2DtpHzQfeNVxAPZgU?si=Jy0f_W0PQKW8eMF1LECenQ) also has a podcast you can find on Spotify that peaked my interest!

[Introduction to Linear Regression in Python](https://towardsdatascience.com/introduction-to-linear-regression-in-python-c12a072bedf0)

[What is One-Hot Encoding and how to use Pandas Get_Dummies Function](https://towardsdatascience.com/what-is-one-hot-encoding-and-how-to-use-pandas-get-dummies-function-922eb9bd4970)

[7 Steps of Machine Learning in practice-a TensorFlow example for structured data](https://towardsdatascience.com/the-googles-7-steps-of-machine-learning-in-practice-a-tensorflow-example-for-structured-data-96ccbb707d77)
