
############  ML in Econometrics ############  
###### Nihal Temüge , 12318000

###### Reference for the analysis: Knaus, M. C. (2022). Double machine learning based program evaluation under unconfoundedness, The Econometrics Journal, forthcoming, arXiv 
# https://mcknaus.github.io/assets/notebooks/appl401k/ANB_401k_GATE.nb.html#GATE_estimation



### set working directory ,  import the dataset , install ruquired packages.

setwd("C:....")
data<-read.csv("genderinequality.csv")
View(data)
install.packages("glmnet")
install.packages("hdm")
library(hdm)
library(glmnet)
library(dplyr)
library(stats)
library(ggplot2)
############################################### WAGES ############################################### 


############  1)	Variable Selection via LASSO Model: ############  


data<-data[data$emp=="1",]
#data<-data[1:18]

data<-na.omit(data)
#data$lnwage<- log(data$wage)
#data<-data[data$lnwage!="-Inf",]


# Assuming 'wage' is the outcome variable
Y <- data$wage

# Create the predictor matrix X
X <- as.matrix(data[, !(names(data) %in% c("id", "year", "wage","treat"))])



lasso_fit <- cv.glmnet(x = X, y = Y, alpha = 1) # alpha=1 for LASSO, alpha=0.5 for elastic net

#After fitting the LASSO model, use cross-validation to identify the optimal value of the lambda (penalty parameter) that minimizes the mean squared error (MSE).
# Get the optimal lambda
optimal_lambda <- lasso_fit$lambda.min

#Model Coefficients with the Optimal Lambda:
#Obtain the model coefficients using the optimal lambda, and check which variables are selected (non-zero coefficients) for variable selection.

# Fit the LASSO model with the optimal lambda
lasso_model <- glmnet(x = X, y = Y, alpha = 1)
#lasso_model <- glmnet(x = X, y = Y, alpha = 1, lambda = optimal_lambda

# Get the coefficients of the selected variables
selected_coefficients <- coef(lasso_model)
selected_variables <- rownames(selected_coefficients)[which(selected_coefficients != 0)]


tmp <- as.data.frame(as.matrix(selected_coefficients))
tmp$coef <- row.names(tmp)
tmp <- reshape::melt(tmp, id = "coef")
tmp$variable <- as.numeric(gsub("s", "", tmp$variable))
tmp$lambda <- lasso_model$lambda[tmp$variable+1] # extract the lambda values
tmp$norm <- apply(abs(selected_coefficients[-1,]), 2, sum)[tmp$variable+1] # compute L1 norm

##### Graph for variables

# x11(width = 13/2.54, height = 9/2.54)
ggplot(tmp[tmp$coef != "(Intercept)",], aes(lambda, value, color = coef, linetype = coef)) + 
  geom_line() + 
  scale_x_log10() + 
  xlab("Lambda (log scale)") + 
  guides(color = guide_legend(title = ""), 
         linetype = guide_legend(title = "")) +
  theme_bw() + 
  theme(legend.key.width = unit(3,"lines"))

################# 2)	Double ML for Augmented Inverse Propensity Score Weighting (AIPW) ################# 


# Load the packages required for later
install.packages("grf")
library(devtools)
install_github(repo="MCKnaus/causalDML")
library(hdm)
library(tidyverse)
library(causalDML)
library(grf)
library(estimatr)


set.seed(1234) # for replicability


# Import dataset, drop NAs, create ln_wage variable, drop "-Inf"s

data<-read.csv("genderinequality.csv")
data$ln_wage<-log(data$wage)
data<- na.omit(data)
data<-data[data$ln_wage!="-Inf",]


# Assign ln_wage as the output variable.
Y = data$ln_wage
# Treatment
W = data$treat
#check selected variables to use them in the following step
selected_variables
# Create main effects matrix by using selected variables with the LASSO analysis in the previous step.

X = model.matrix(~ 0 + hours + female + IQ + KWW + educ + exper + tenure + age +married +black + south + urban +sibs +brthord +meduc +feduc , data = data)

# 5-fold cross-fitting with causalDML package
aipw = DML_aipw(Y,W,X)
summary(aipw$ATE)



# tune the forest

forest = create_method("forest_grf",args=list(tune.parameters = "all"))
aipw = DML_aipw(Y,W,X,ml_w=list(forest),ml_y=list(forest),cf=5)

#GATE estimation
#The pseudo-outcome can now be used to estimate different heterogeneous effects. I use standard regression models but by using the pseudo-outcome instead of a real outcome we model effect size and not outcome level.

# Subgroup effect
# Check Gender differences: This would usually be implemented by splitting the sample by gender and rerunning the whole analysis in the subsamples separately.
# 
# With the pseudo-outcome stored in aipw$ATE$delta this boils down to running an OLS regression with the female indicator as single regressor.
# 


female = X[,2]
blp_female = lm_robust(aipw$ATE$delta ~ female)
summary(blp_female)



# the gender specific effect instead of differences between groups, just run an OLS regression without constant and all group indicators:

male = 1-female
blp_female1 = lm_robust(aipw$ATE$delta ~ 0 + male + female)
summary(blp_female1)

# I can transfer all the strategies that we know about modeling outcomes with OLS for modelling causal effects.


# Best linear prediction
#  I do not want to focus on subgroup analyses but to model the effect using all main effects at our disposal. In standard OLS this would mean to include a lot of interaction effects while completely relying on correct specification of the outcome model.
# 
# Using the pseudo-outcome allows us to be completely agnostic about the outcome model and to receive a nice summary of the underlying effect heterogeneity in a familiar format, an OLS output:
#   
blp = lm_robust(aipw$ATE$delta ~ X)
summary(blp)

# Non-parametric heterogeneity
# having the pseudo-outcome . can also estimate heterogeneous effects with nonparametric regressions. This means we are not only agnostic about the outcome and propensity score models but also about the functional of effect heterogeneity.
# 
# This is especially useful if we have some continuous variable like age for which we want to understand effect heterogeneity.

##########  EDUCATION ##########  
## Spline regression

educ = X[,5]
sr_educ = spline_cate(aipw$ATE$delta,educ)

plot(sr_educ,z_label = "Education")


## Kernel Regression 

kr_educ = kr_cate(aipw$ATE$delta,educ)
plot(kr_educ,z_label = "Education")

##########  AGE ##########
## Spline regression

age = X[,8]
sr_age = spline_cate(aipw$ATE$delta,age)

plot(sr_age,z_label = "Age")


## Kernel Regression 

kr_age = kr_cate(aipw$ATE$delta,age)
plot(kr_age,z_label = "Age")

##########  IQ ##########
## Spline regression

iq = X[,3]
sr_iq = spline_cate(aipw$ATE$delta,iq)

plot(sr_iq,z_label = "IQ")


## Kernel Regression 

kr_iq = kr_cate(aipw$ATE$delta,iq)
plot(kr_iq,z_label = "IQ")




###########################################################  EMPLOYMENT ###########################################################  

### import data , drop NAs

data<-read.csv("genderinequality.csv")
data<- na.omit(data)



############  1)	Variable Selection via LASSO Model: ############  

# Assuming 'emp' is the outcome variable
Y <- data$emp

# Create the predictor matrix X
X <- as.matrix(data[, !(names(data) %in% c("id", "year", "emp"))])



lasso_fit <- cv.glmnet(x = X, y = Y, alpha = 1) # alpha=1 for LASSO, alpha=0.5 for elastic net

#After fitting the LASSO model, use cross-validation to identify the optimal value of the lambda (penalty parameter) that minimizes the mean squared error (MSE).
# Get the optimal lambda
optimal_lambda <- lasso_fit$lambda.min

#Model Coefficients with the Optimal Lambda:
#Obtain the model coefficients using the optimal lambda, and check which variables are selected (non-zero coefficients) for variable selection.

# Fit the LASSO model with the optimal lambda

lasso_model <- glmnet(x = X, y = Y, alpha = 1, lambda = optimal_lambda)

# Get the coefficients of the selected variables
selected_coefficients <- coef(lasso_model)
selected_variables <- rownames(selected_coefficients)[which(selected_coefficients != 0)]


################# 2)	Double ML for Augmented Inverse Propensity Score Weighting (AIPW) ################# 


W = data$treat


selected_variables
# Create main effects matrix
X = model.matrix(~ 0 + wage+hours + female + IQ + KWW + educ + exper + tenure + age +married +black + south + urban +sibs +brthord +meduc +feduc , data = data)

# 5-fold cross-fitting with causalDML package
aipw = DML_aipw(Y,W,X)
summary(aipw$ATE)



# tune the forest

forest = create_method("forest_grf",args=list(tune.parameters = "all"))
aipw = DML_aipw(Y,W,X,ml_w=list(forest),ml_y=list(forest),cf=5)

#GATE estimation
#The pseudo-outcome can now be used to estimate different heterogeneous effects. We use standard regression models but by using the pseudo-outcome instead of a real outcome we model effect size and not outcome level.
#


# Subgroup effect: Gender differences

female = X[,3]
blp_female = lm_robust(aipw$ATE$delta ~ female)
summary(blp_female)



# the gender specific effect instead of differences between groups, just run an OLS regression without constant and all group indicators:

male = 1-female
blp_female1 = lm_robust(aipw$ATE$delta ~ 0 + male + female)
summary(blp_female1)


# Using the pseudo-outcome allows us to be completely agnostic about the outcome model and to receive a nice summary of the underlying effect heterogeneity in a familiar format, an OLS output:
#   
blp = lm_robust(aipw$ATE$delta ~ X)
summary(blp)


# Non-parametric heterogeneity

##########  EDUCATION ##########  
## Spline regression

educ = X[,6]
sr_educ = spline_cate(aipw$ATE$delta,educ)

plot(sr_educ,z_label = "Education")


## Kernel Regression 

kr_educ = kr_cate(aipw$ATE$delta,educ)
plot(kr_educ,z_label = "Education")

##########  AGE ##########
## Spline regression

age = X[,9]
sr_age = spline_cate(aipw$ATE$delta,age)

plot(sr_age,z_label = "Age")


## Kernel Regression 

kr_age = kr_cate(aipw$ATE$delta,age)
plot(kr_age,z_label = "Age")
kr_age$ate
sr_age$ate
# title (main= "Kernel Regression(employment",line=NA, outer = F)


























