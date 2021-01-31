# COVID19-prediction

## Q.1. Which model have you used for Covid Cases prediction? Explain your model.

In this task (part 1), we have used a multilayer perceptron neural network. In this task we have mainly focused on cleaning the data and *dimensionality reduction* through the trends of every parameter with each other and also with the Covid Cases parameter. 

### Data Exploration / Feature Engineering:

Some of the conclusions of our data exploration:
1.	By using correlation method in pandas we have concluded that Sex ratio, Median Age, Avg Temp, Water purity and H index are least significantly correlated to Covid Cases and so we have omitted these parameters.

2.	Now from the rest of the parameters it has been observed that Female population and Population [2001], are highly correlated to Population [2011], so from here it can be inferred that these parameters are a sub parameter of Population [2011], hence have been dropped.

3.	In this task we have 4 categorical data City, State, Type and SWM. We have dropped all 4 of them:
a.	City:  Too many values to be categorised
b.	State: On using boxplot of every state vs covid cases and state vs population, it was inferred that the median is almost the same for every state, which means irrespective of your state, if the population is high the city will have a high rate of covid cases.
c.	SWM: On visualising the boxplot of SWM vs covid cases, it was inferred that for each category high, low or medium the median, mean, max value, min value, mean, 25% data , 75% data, all were the same. So irrespective of which category of SWM your city belongs to, the covid cases will not explicitly depend on the SWM category.

d.	Type: We observed in type vs coid cases and type vs population, that the median is more for those types whose population is also more. So from here we concluded that type’s effect would be covered in the Population [2011].

### Handling Missing Data:

For handling of missing data, we have used an advanced Imputation technique called **Multivariate Imputation by Chained Equations(MICE)**. Multiple imputation has a number of advantages over these other missing data approaches. Multiple imputation involves filling in the missing values multiple times, creating multiple “complete” datasets.
In the MICE procedure a series of regression models are run whereby each variable with missing data is modeled conditional upon the other variables in the data. This means that each variable can be modeled according to its distribution, with, for example, binary variables modeled using logistic regression and continuous variables modeled using linear regression.
This  was made possible using IterativeImputer library of sklearn.impute.



### Multilayer Perceptron Neural Network:

We have made a **4 layer network network with 256 hidden units** in each layer, and used adam optimiser, mean squared error and root mean squared error as metric. We have used 2000 epochs, split the data into 90%-10% train-test data and used 10% of training data as validation data. Batch size of 64 has been selected in the model. Relu activation has been used in the hidden units.









## Q.2.	Which model have you used for Foreign Visitors Time series prediction? Explain your model.

In our final model we have used an **LSTM layer** (of 2 units) followed by a Dense layer (3 units) and then a final Dense output neuron (1 unit). 

Since we did not have any training data from October, we framed the model as a Time Series Forecasting problem where the input was data from 4 months and we predict the next month. We trained on the given data by generating the inputs and outputs in this way, and then used it to predict the no of Foreign Visitors in October. Since we didn’t have the data to check for *overfitting*, we had to ensure that we don’t increase the complexity of the models for a marginal increase in accuracy.

We tried different types of architectures starting from a baseline where visitors in September = visitors in August. The second was a Multi Layer Perceptron based model where we tried a model with two Densely connected layers(x units and 1 unit respectively, where x was a hyperparameter to tune). Finally we tried different combinations of LSTM models and tuned the no of LSTM units and Dense units to arrive at the final model which gave a **MSE of 0.004 and MAE of 0.03**. 


