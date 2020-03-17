

#Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)


#  - Importing Essential Packages and Modules

# 1 - DMwR - Allows you to balance the data set using sampling methods (SMOTE).
# 2 - caret - Provides tools for machine learning activities.
# 3 - scales - Scaling and Centering of Matrix-like Objects.
# 4 - randomForest -  mplements Breiman's random forest algorithm for classification and regression.
# 5 - ggplot2 - Create Elegant Data Visualisations Using the Grammar of Graphics.
# 6 - dplyr - Provides tools for manipulating datasets.
# 7 - ada - Performs discrete, real, and gentle boost for classification.
# 8 - forcats - Reorder Factor Levels By Sorting Along Another Variable.
# 9 - ROCR - Visualizing the Performance of Scoring Classifiers.

#If you don't have any of these packages, described above, already insttalled in your rstudio
#please, run the code below!

packs <- c('DMwR', 'caret', 'scales', 'randomForest', 'ggplot2', 'dplyr', 'ada', 'forcats', 'ROCR')
install.packages(packs)

#If you have some of the packages, please run the code below giving the name of the package that are missing for you!

install.packages('Your missing package')


#Loading

lapply(packs, require, character.only = T)

# The Data

df <- read.csv('credit_dataset.csv', header = T, sep = ',', stringsAsFactors = F)

str(df)
View(df)

# Feature Engineering

# Adding a new column with age groups

summary(df$age)
 
df["age_categ"] <- cut(df$age, breaks = c(15, 30, 45, 60, 75))

# Creating a function to convert variables into factors.

tofactor <- function(dataf, var){
  for (variable in var){
    dataf[[variable]] <- as.factor(dataf[[variable]])
  }
  return(dataf)
}


# Creating a function to normalize the numerical variables.

normalizing <- function(dataf, var){
  for(variable in var){
    dataf[[variable]] <- scale(dataf[[variable]], center = T, scale = T)
  }
  return(dataf)
}

# Spliting columns in categoricals and numericals.

categ_vars <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                'other.credits', 'apartment.type', 'bank.credits', 'occupation', 
                'dependents', 'telephone', 'foreign.worker', 'age_categ')

numeric_vars <- colnames(df[, - match(categ_vars, names(df))])


#Transforming the dataframe.

df <- normalizing(df, numeric_vars)

# Converting to factor
df <- tofactor(df, categ_vars)

str(df)


# Checking Imbalance Data

# Numerical Approach

table(df$credit.rating)

# Grapical Approach

df %>% 
  ggplot(aes(x = credit.rating, fill = credit.rating)) +
  geom_bar(aes(y = (..count..)/sum(..count..))) +
  geom_text(aes(y = ((..count..)/sum(..count..)), 
                label = percent((..count..)/sum(..count..))), 
                stat = "count", vjust = -0.25) +
  theme(plot.title = element_text(face = "bold")) +
  labs(title="Checking Imbalance Data ", subtitle="analysing the dependent variable",
       caption = 'Source: Data collected from German Credit Data Set', 
       y="Frequencies", x="Credit Rating") +
  scale_x_discrete(breaks = c("0", "1"),
                   labels = c("Bad", "Good")) +
  scale_fill_discrete(name = "Rating", labels = c("Bad", "Good")) +
  scale_y_continuous(labels = percent) 


# it is proven that the data set is unbalanced, so we have to apply sampling Methods 
# to Balance the Data avoiding a tendentious ML model.
# The chosen method of sampling was the Synthetic Minority Over-Sampling Technique (SMOTE).
#This method is used to avoid overfitting when adding exact replicas of minority instances to the main dataset.

set.seed(123)
newdf <- SMOTE(credit.rating ~ ., df, perc.over = 100, perc.under = 200)

table(newdf$credit.rating)

colnames <- colnames(newdf)

# Exploring the data

grid.names <- c('BAD', 'GOOD')
names(grid.names) <- c(0, 1)

lapply(colnames, function(x){
  if(is.factor(newdf[,x])) {
    ggplot(newdf, aes_string(x)) +
      geom_bar() + 
      facet_grid(. ~ credit.rating, labeller = labeller(credit.rating = grid.names)) + 
      labs(title="Credit Risk", subtitle= paste("by column", x),
           caption = 'Source: Data collected from German Credit Data Set', 
           y="Frequencies")}})

# Feature selection using Random Forest
set.seed(123)
rmodel <- randomForest(credit.rating ~ ., data = newdf, ntree = 150, nodesize = 15, importance = T)

# Feature selection using Learning Vector Quantization (LVQ) model
set.seed(123)
control.lvq <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)

lvq.model <- train(credit.rating ~ ., data = newdf, method='lvq', preProcess="scale", trControl=control.lvq)

# Feature selection using Recursive Feature Elimination or RFE
set.seed(123)
control.rfe <- rfeControl(functions=rfFuncs, method="cv", number=10)
results <- rfe(newdf[,2:22], newdf[,1], sizes=c(1:8), rfeControl=control.rfe)

-------------------------------------------------------------------------------------------

  
# Evaluating the features using Filter Based Feature Selection from ML Azure.

# according to rfe model the 5 best features should be:

# account.balance, previous.credit.payment.status, credit.duration.months, savings, other.credits  
-----------  
  
# Evaluating the features using random forest importance plot.

varImpPlot(rmodel)

# according to rmodel the 5 best features should be:

# account.balance, credit.duration.months, credit.amount, previous.credit.payment.status, other.credits 

-----------
# Evaluating the Learning Vector Quantization (LVQ) model  

# Importance Table
varImp(lvq.model)

# Importance Charts
ggplot(varImp(lvq.model)) +
  theme(plot.title = element_text(face = "bold")) +
  labs(title="Feature Selection", subtitle=" by Learning Vector Quantization (LVQ)",
       caption = 'Source: Data collected from German Credit Data Set', 
       y="Importance", x="Features") 

# according to lvq.model the 5 best features should be:

# account.balance, credit.duration.months, credit.amount, previous.credit.payment.status, other.credits

-----------
  
# Evaluating the Recursive Feature Elimination or RFE 

# Importance Table 
varImp(results)

# Importance Charts

rfe.df <- data.frame(varImp(results))
rfe.df <- cbind(rfe.df, row.names(rfe.df))
colnames(rfe.df) <- c('Importance','Features')
rfe.df$Features <- factor(rfe.df$Features)

rfe.df %>%
    mutate(Features = fct_reorder(Features, Importance)) %>% 
    ggplot(aes(x = Features , y = Importance)) +
      geom_bar(stat = 'identity') +
      coord_flip() +
      theme(plot.title = element_text(face = "bold")) +
      labs(title="Feature Selection", subtitle=" by Recursive Feature Elimination (RFE)",
           caption = 'Source: Data collected from German Credit Data Set', 
           y="Importance", x="Features")
     


# according to rfe model the 5 best features should be:

# account.balance, credit.duration.months, credit.amount, previous.credit.payment.status, credit.purpose



# We're going to use ML Azure Studio to test different machine learning models using
# the best features from each method.
# the one who gets the highest score will be chosen as our model.

# RESULTS FROM ML AZURE STUDIO
  
# Features Selected By Azure

# - Boosted Decision Tree: 0.765
# - SVM: 0.735
# - Neural Network: 0.738 
# - Logistic Regression: 0.750

# Features Selected By Random Forest

# - Boosted Decision Tree: 0.773
# - SVM: 0.723
# - Neural Network: 0.731
# - Logistic Regression: 0.738

# Features Selected By LVQ

# - Boosted Decision Tree: 0.769
# - SVM: 0.712
# - Neural Network: 0.727
# - Logistic Regression: 0.746

# Features Selected By RFE

# - Boosted Decision Tree: 0.773
# - SVM: 0.723
# - Neural Network: 0.731
# - Logistic Regression: 0.738
  
  
# Based on the results above, we're going to implement the best two models using RFE featuring Selection:
# - Boosted decion tree and Logistic Regression.


formula <- as.formula('credit.rating ~ account.balance + credit.duration.months + 
                      previous.credit.payment.status + credit.purpose + credit.amount')


# Partitioning the data into train and test 80 - 20.

set.seed(123)
idx <- createDataPartition(newdf$credit.rating, p = .8, list = F, times = 1)

df.train <- newdf[idx,]
df.test <- newdf[-idx,]

# Training and evaluating the previous two best models


# Training a  Additive Logistic Regression.
set.seed(123)
ada.model <- ada(formula, data = df.train, type='gentle')


#Predictions for Additive Logistic Regression 
ada.pred <- predict(ada.model, df.test)
ada.pred.t <- as.numeric(as.character(ada.pred))

#Evaluation for Additive Logistic Regression 

# Confusion Matrix for Additive Logistic Regression
confusionMatrix(table(ada.pred, df.test$credit.rating), positive = '1')

# ROCR Curve for Additive Logistic Regression

set.seed(123)
ada.obj <- prediction(ada.pred.t, df.test$credit.rating) 
ada.performance <- performance(ada.obj, 'tpr', 'fpr')

plot(ada.performance, colorize = T, lty = 1, lwd = 3, 
     main = "ROC Curve")
abline(0,1, col = "black")
auc <- performance(ada.obj, "auc")
auc <- unlist(slot(auc, "y.values"))
auc <- round(auc,2)
legend(0.4,0.4, legend = c(paste0("AUC: ",auc)), cex = 0.6, bty = "n", box.col = "white")

---------------------------------------------------------------------------------------------
  
# Training a Boosted Decision Tree.
  
set.seed(123)
xgb.model <- train(formula, data = df.train, method = 'xgbTree')

# Predictions for Boosted Decision Tree.

xgb.pred <- predict(xgb.model,df.test)
xgb.pred.t <- as.numeric(as.character(xgb.pred))


# Confusion Matrix for Boosted Decision Tree.

confusionMatrix(table(xgb.pred, df.test$credit.rating), positive = '1')

#ROCR Curve for Boosted Decision Tree.

set.seed(123)
xgb.obj <- prediction(xgb.pred.t,df.test$credit.rating)
xgb.performance <- performance(xgb.obj, 'tpr', 'fpr')

plot(xgb.performance, colorize = T, lty = 1, lwd = 3, 
     main = "ROC Curve")
abline(0,1, col = "black")
auc <- performance(xgb.obj, "auc")
auc <- unlist(slot(auc, "y.values"))
auc <- round(auc,2)
legend(0.4,0.4, legend = c(paste0("AUC: ",auc)), cex = 0.6, bty = "n", box.col = "white")



# The Prime Model

plot(ada.performance, col = 'blue', lty = 1, lwd = 3, 
     main = "ROC Curve")
plot(xgb.performance, add = T, col = 'red', lty = 1, lwd = 3, 
     main = "ROC Curve")
legend(0.2,0.7, legend = c('ADA.MODEL'), cex = 0.6, bty = "p", box.col = "blue")
legend(0.05,0.6, legend = c('XGB.MODEL'), cex = 0.6, bty = "p", box.col = "red")

# So the boosted decion tree has the best performance, so we're going to select it and try to
# run some sort of optimization process.

# Trying to Optmize our best model

set.seed(123)
control <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)

model <- train(formula, data = df.train, method='xgbTree', trControl = control)

pred <- predict(model, df.test)

confusionMatrix(table(pred, df.test$credit.rating), positive = '1')


























