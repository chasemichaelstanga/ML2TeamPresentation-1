rm(list=ls())
install.packages("glmnet")
install.packages("splines")
install.packages("readxl")
library(readxl)
library(glmnet)
library(splines)

set.seed(5082)
my.df<-read.csv(file="zillow2223.csv", header = T, sep = ",", stringsAsFactors = TRUE, row.names = 1)

my.df$zipcode<-as.factor(my.df$zipcode)
my.df$age<-2023-my.df$year
#my.df<-cbind(my.df,age)
print(summary(my.df$age))

my.df<-subset(my.df,select=-c(section, team, international, masonyear, zestimate, assessment, taxes, willingtopay, year))
attach(my.df)


# putting the parameters/predictors into a matrix 'x'
x<-model.matrix(price~., my.df)[,-1]
# putting the Y, or the variable we are trying to predict into the vector 'y'
y<-my.df$price

'sample(x, size, replace = FALSE, prob = NULL)'

trainIndex<-sample(1:nrow(x), nrow(x)*.8)
#test <-(!trainIndex)

train.x <- x[trainIndex,]
train.y <- y[trainIndex]

test.x <- x[-trainIndex,]
test.y <- y[-trainIndex]

print(test.y)
print(summary(test.y))
#################
'''
Regularization is a method of minimizing the important of Features, or Feature Reduction. 
 

Regularization adjusts the Beta coefficients (weights) associated with
each feature (column).

Ridge and Lasso. 


Lasso is when you are able to fully remove the effect of columns by 
reducing the weights all the way down to 0. Multiplying a column by 0 
makes it disappear from the model. 
      It is most suitable when a data set contains a higher number or observations than the number or predictor variables

Ridge is when you reduce the weights of the different columns, but never
make them 0. In Ridge, each column will still have some weight, even if
it is really low like 0.0001. 
      It is most suitable when a data set contains a higher number of predictor variables than the number of observations.

'''
'''
Two other notes about Regularization.
1: Unlike weve used for other models, this requires an x matrix and a
separate y matrix to work. Normally, we have all the data in one df and
use y ~ x. For any glmnet function we have to use matrix format.


'''


#     Print beta coefficients of the model associated with the best lambda. (#Q02-4)
set.seed(5082)
options(scipen= 999)

power.value<-seq(from=4, to=-2, length=150)
power.value

grid<-10^power.value
grid
 

# A new term Lambda refers to how strongly we allow Regularization to take
# place. The higher the Lambda, the smaller the weights. A higher lambda 
# means that we are removing more of the data, so smaller weights.
# We often dont know the best Lambda to use, so we feed in a list of 
# potential values and let the code try all of them. Then we pick the best
# one after the model runs.

# Another new term Alpha that dictates how much we use Ridge vs how much 
# we use Lasso. 
# 
# Alpha = 0 is exclusively Ridge
# 
# Alpha = 1 is exclusively Lasso

mod.lasso<-glmnet(x=x.train, y=y.train, alpha=1, lambda=grid)

cv.out.lasso<-cv.glmnet(x=train.x, y=train.y, alpha=1, lambda=grid, nfolds = 12) #may or may not specify folds
bestlam <- cv.out.lasso$lambda.min
print(bestlam)

#plot(cv.out.lasso$lambda)
#print(grid)

lasso.pred<-predict(cv.out.lasso, s=bestlam, newx = test.x)
mean((lasso.pred-test.y)^2)

print(lasso.pred)
print(mean((lasso.pred-test.y)^2))

#out<-glmnet(x,y, alpha=1, lambda=grid)
lasso.coef<-predict(mod.lasso, type="coefficients", s=bestlam)
print(lasso.coef)



################# 3 ##################

######## ClASSSIFICATION EXAMPLE ###########
set.seed(5082)
x = model.matrix(zipcode~., my.df)[, -1]
y<-my.df$zipcode
n <- nrow(my.df)
trainIndex <- sample(n, .8 * n)
train.x <- x[trainIndex, ]
test.x <- x[-trainIndex, ]
train.y <- my.df$zipcode[trainIndex]
test.y <- my.df$zipcode[-trainIndex]

mod.ridge <- glmnet(train.x, train.y, alpha=0,family = "binomial")

cv.out.ridge <- cv.glmnet(train.x, train.y, alpha=0, lambda=grid,nfolds=12,family = "binomial")

bestlam1.ridge <- cv.out.ridge$lambda.min

bestlam1.ridge

ridge.pred <- predict(mod.ridge, s=bestlam1.ridge, newx=test.x, type = "response") # or it can be 'class'

print(ridge.pred) # 
# response = probabilities
# class = the class
ridge.coef<-predict(mod.ridge, type="coefficients", s=bestlam1.ridge)
ridge.coef

# Print the test set predictions in terms of zipcode (i.e., 23185, or 23188 - no quotation marks). 
# Use contrasts() to determine which zipcode is the positive class. Use >= .5 as the classification rule. 
# The output should be a table with two columns: the first is a list of student names, and the second is 
# a list of zipcodes. (#Q03-4)
print(ridge.coef)
contrasts(train.y) # "predicting whether a house is in 23188, because it is represented by 1

class.pred <- ifelse(ridge.pred >= .5, 23188, 23185)
class.pred
# the probability given when type = "response" is the probability the obsv is in 23188 since 23185 is the referent category

# Print a list of TRUE/FALSE values indicating whether each test prediction is correct (TRUE) or not (FALSE). 
# The output should be a table with two columns: the first is a list of student names, and the second is a 
# list of TRUE/FALSE values. (#Q03-5)

tf.pred <- ifelse(class.pred == test.y, T, F)
tf.pred
#Print overall model accuracy rate of the test set predictions (with 3 decimal values; not percentage %). (#Q03-6)
mean(class.pred == test.y)

################## 4 ####################

# Include all of the code from the previous questions. Do NOT comment out print statements.
# Set seed to 5082.

# Create a vector of 8 elements, filled with zeros. Name this vector ns.cv.mse. We will use it to store cross-validation MSEs.
# Train a cubic Natural Splines model to predict price, and evaluate its cross-validation test performance.

# i. Use sqft as a predictor, with the training dataset.
# ii. Compare test MSEs of 1 to 8 knots, with a for loop, and the test dataset. Be sure to use price, not zipcode as the target in test.
# iii. Store the cross-validation errors into ns.cv.mse.
# Print ns.cv.mse. (#Q04-1)
#   In the RStudio window, plot the cross-validation mses by number of knots. Choose the best model using the 1-SD rule, based on the plot.
set.seed(5082)

test.y <- my.df$price[-trainIndex]

ns.cv.mse=seq(from=0, to=0, length.out=8)
for (i in 1:length(ns.cv.mse)){
NSmodel <- glm(price~ns(sqft, ), data=my.df[trainIndex,])
}
ns.pred <- predict(NSmodel, newdata=my.df[-trainIndex,])
ns.cv.mse[1] <- mean((ns.pred - test.y)^2)
print(ns.cv.mse)


plot(ns.cv.mse[1:8])

(min <- which.min(ns.cv.mse))
best.knots=6

BestNSModel <- lm(price~ns(sqft, df=3+ best.knots -2), data=my.df[trainIndex,])
price_hat_ns <- predict(BestNSModel, newdata=my.df[-trainIndex,])
price_hat_ns










