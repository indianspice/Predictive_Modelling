library(psych)
library(GGally)
library(ggplot2)
library(ggthemes)
library(reshape2)
#library(VIM)
# library(mice)
# library(stringr)
library(dplyr)
library(car)
library(usdm)
library(tidyverse)
library(stringr)
library(DataExplorer)
library(knitr)
# library(corrplot)
library(MASS)
library(Metrics)
# library(tinytex)
# library(ggfortify)
library(caret)
library(pscl)
library(MKmisc)
library(Metrics)
library(pROC)
library(rpart)
library(rpart.plot)
library(glmnet)
library(neuralnet)

# library(gvlma)  ## only used for confirming model assumptions

options(scipen=999)


## ----Read data, echo=FALSE, message=FALSE, warning=FALSE-----------------

cc_data <- read.csv("https://raw.githubusercontent.com/621-Group2/Final-Project/master/UCI_Credit_Card.csv", header=TRUE, sep=",")

#Remove the id from the dataset
cc_data$ID <- NULL

#Simplify name of response
colnames(cc_data)[24] <- "DEFAULT" 


## ----user_functions, echo=FALSE, message=FALSE, warning=FALSE------------

as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}

# get_outliers function
get_outliers <-  function(x, n = 10) {
  
  bp <- boxplot.stats(x)
  
  obs_hi <- unique(x[which(x > bp$stats[5])])

  if (length(obs_hi) < n) { n <- length(obs_hi) }

  hi <- sort(obs_hi, decreasing = T)[1:n]
  
  obs_low <- unique(x[which(x < bp$stats[1])])

  if (length(obs_low) < n) { n <- length(obs_low) }

  low <- sort(obs_low, decreasing = T)[1:n]

  return (list(Hi=hi, Low=low))
  
}  

#Keith's function
all_model_metrics <- data.frame()
all_roc_curves <- list()
all_predictions <- list()
all_cm <- list()

calc_metrics <- function(model_name, 
                         model, 
                         test, 
                         train) {
  
  pred_model <- predict(model, test, type = 'response')
  y_pred_model <- factor(ifelse(pred_model > 0.5, 1, 0), levels=c(0, 1))

  compare <- cbind (actual=as.numeric.factor(test$target), 
                    predicted=as.numeric.factor(y_pred_model))  # combine
  
  # Confusion Matrix
  cm <- confusionMatrix(test$target, y_pred_model, positive = "1", mode="everything" ) 
  
  accuracy <- cm$overall[[1]]
  kappa_value <- cm$overall[[2]]
  youden_value <- cm$byClass[[1]] - (1 - cm$byClass[[2]])
  F1Score_value <- cm$byClass[[7]]
  FP_value <- (cm$table[2,1]/nrow(test))*100
  
  #AUC
  #AUC_value <- auc(as.numeric(test$target), pred_model)
  
  cm_df <- data.frame(Model=model_name, 
                      Accuracy=round(accuracy, 3),
                      AIC=round(AIC(model), 3), 
                      BIC=round(BIC(model), 3), 
                      MAE=Metrics::mae(compare[, 1], compare[, 2]),
                      MAPE=Metrics::mape(compare[, 1], compare[, 2]),     #MAPE calculates the mean absolute percentage error:
                      MSE=Metrics::mse(compare[, 1], compare[, 2]),       #MSE calculates mean squared error
                      RMSE=Metrics::rmse(compare[, 1], compare[, 2]),
                      Kappa = round(kappa_value, 3), 
                      Youden = round(youden_value, 3), 
                      F1Score = round(F1Score_value, 3),
                      FPPrct = round(FP_value, 2),
                      Sensitivity=round(cm$byClass["Sensitivity"], 3),
                      Specificity=round(cm$byClass["Specificity"], 3),
                      PosPredValue=round(cm$byClass["Pos Pred Value"], 3),
                      NegPredValue=round(cm$byClass["Neg Pred Value"], 3),
                      Precision=round(cm$byClass["Precision"], 3),
                      Recall=round(cm$byClass["Recall"], 3),
                      BalancedAccuracy=round(cm$byClass["Balanced Accuracy"], 3)
                      )
  
  #cbind(t(cm$overall),t(cm$byClass)))
  
  # ROC Curves 
  roc_model <- roc(target ~ pred_model, data = test)
  
  # Result
  result <- list(cm_df, roc_model, compare, cm)

  return (result)
  
}

## this for used for the ridge and lasso regression models
calc_metrics_2 <- function(model_name, 
                           model, 
                           test, 
                           train, 
                           s,
                           y_test) {
  
    
  pred_model <- predict(model, newx = test, s=s, type="response")
  y_pred_model <- as.factor(ifelse(pred_model > 0.5, 1, 0))
     
  compare <- cbind (actual=as.numeric.factor(y_test), 
                    predicted=as.numeric.factor(y_pred_model))  # combine
  
  # Confusion Matrix
  cm <- confusionMatrix(as.factor(y_test), y_pred_model, positive = "1", mode="everything" ) 
  
  accuracy <- cm$overall[[1]]
  kappa_value <- cm$overall[[2]]
  youden_value <- cm$byClass[[1]] - (1 - cm$byClass[[2]])
  F1Score_value <- cm$byClass[[7]]
  FP_value <- (cm$table[2,1]/nrow(test))*100
  
  #AUC
  #AUC_value <- auc(as.numeric(test$target), pred_model)
  
  cm_df <- data.frame(Model=model_name, 
                      Accuracy=round(accuracy, 3),
                      AIC=NA, 
                      BIC=NA, 
                      MAE=Metrics::mae(compare[, 1], compare[, 2]),
                      MAPE=Metrics::mape(compare[, 1], compare[, 2]),     #MAPE calculates the mean absolute percentage error:
                      MSE=Metrics::mse(compare[, 1], compare[, 2]),       #MSE calculates mean squared error
                      RMSE=Metrics::rmse(compare[, 1], compare[, 2]),
                      Kappa = round(kappa_value, 3), 
                      Youden = round(youden_value, 3), 
                      F1Score = round(F1Score_value, 3),
                      FPPrct = round(FP_value, 2),
                      Sensitivity=round(cm$byClass["Sensitivity"], 3),
                      Specificity=round(cm$byClass["Specificity"], 3),
                      PosPredValue=round(cm$byClass["Pos Pred Value"], 3),
                      NegPredValue=round(cm$byClass["Neg Pred Value"], 3),
                      Precision=round(cm$byClass["Precision"], 3),
                      Recall=round(cm$byClass["Recall"], 3),
                      BalancedAccuracy=round(cm$byClass["Balanced Accuracy"], 3)
                      )
  
  #cbind(t(cm$overall),t(cm$byClass)))
  
  # ROC Curves 
  roc_model <- roc(as.numeric(y_test) ~ as.numeric(pred_model), data = as.data.frame(test))
  
  # Result
  result <- list(cm_df, roc_model, compare, cm)

  return (result)
  
}

## used for neuralnet and decision tree
calc_metrics_3 <- function(model_name, 
                           model, 
                           test, 
                           train,
                           type=NULL,
                           y_test=NULL) {
  
  if (type=="nnet") {
    pred_model <- neuralnet::compute(model, test[, -23])$net.result
    y_pred_model      <- as.factor( ifelse(pred_model > 0.5, 1, 0) )
    target <- y_test
  }
  else { 
    
    pred_model <- predict(model, test, type = "prob")[,"1"]
    y_pred_model      <- as.factor(ifelse(pred_model > 0.5, 1, 0))
    target <- test$target
  }
  
  
  compare <- cbind (actual=as.numeric.factor(target), 
                    predicted=as.numeric.factor(y_pred_model))  # combine
  
  # Confusion Matrix
  cm <- confusionMatrix(as.factor(target), y_pred_model, positive = "1", mode="everything" ) 
  
  accuracy <- cm$overall[[1]]
  kappa_value <- cm$overall[[2]]
  youden_value <- cm$byClass[[1]] - (1 - cm$byClass[[2]])
  F1Score_value <- cm$byClass[[7]]
  FP_value <- (cm$table[2,1]/nrow(test))*100
  
  #AUC
  #AUC_value <- auc(as.numeric(test$target), pred_model)
  
  cm_df <- data.frame(Model=model_name, 
                      Accuracy=round(accuracy, 3),
                      AIC=NA, 
                      BIC=NA, 
                      MAE=Metrics::mae(compare[, 1], compare[, 2]),
                      MAPE=Metrics::mape(compare[, 1], compare[, 2]),     #MAPE calculates the mean absolute percentage error:
                      MSE=Metrics::mse(compare[, 1], compare[, 2]),       #MSE calculates mean squared error
                      RMSE=Metrics::rmse(compare[, 1], compare[, 2]),
                      Kappa = round(kappa_value, 3), 
                      Youden = round(youden_value, 3), 
                      F1Score = round(F1Score_value, 3),
                      FPPrct = round(FP_value, 2),
                      Sensitivity=round(cm$byClass["Sensitivity"], 3),
                      Specificity=round(cm$byClass["Specificity"], 3),
                      PosPredValue=round(cm$byClass["Pos Pred Value"], 3),
                      NegPredValue=round(cm$byClass["Neg Pred Value"], 3),
                      Precision=round(cm$byClass["Precision"], 3),
                      Recall=round(cm$byClass["Recall"], 3),
                      BalancedAccuracy=round(cm$byClass["Balanced Accuracy"], 3)
                      )
  
  #cbind(t(cm$overall),t(cm$byClass)))
  
  # ROC Curves 
  roc_model <- roc(as.numeric(target) ~ as.numeric(pred_model), data = as.data.frame(test))
  
  # Result
  result <- list(cm_df, roc_model, compare, cm)

  return (result)
  
}

plot_varImp <- function(model) {
  
  x <- data.frame(varImp(model))

  x$Variable <- rownames(x)

  x %>% ggplot(aes(x=reorder(Variable, Overall), y=Overall, fill=Overall)) +
            geom_bar(stat="identity") + coord_flip() + guides(fill=FALSE) +
            xlab("Variable") + ylab("Importance") + 
            ggtitle("Variable Importance") 
  
  
}



## ----data exploration, echo=FALSE, message=FALSE, warning=FALSE----------

Variable <- colnames(cc_data)

Definition <- c("Amount of given credit in NT dollars", "Gender (1=male, 2=female)", "(1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)", "Marital status (1=married, 2=single, 3=others)", "Age in Years", "Repayment status in September, 2005", "Repayment status in August, 2005", "Repayment status in July, 2005", "Repayment status in June, 2005", "Repayment status in May, 2005", "Repayment status in April, 2005", "Amount of bill statement in September, 2005", "Amount of bill statement in August, 2005", "Amount of bill statement in July, 2005", "Amount of bill statement in June, 2005", "Amount of bill statement in May, 2005", "Amount of bill statement in April, 2005", " Amount of previous payment in September, 2005", " Amount of previous payment in August, 2005", " Amount of previous payment in July, 2005", " Amount of previous payment in June, 2005", " Amount of previous payment in May, 2005", " Amount of previous payment in April, 2005", "Default payment (1=yes, 0=no)")

card_sum <- cbind.data.frame (Variable, Definition)

card_sum$Type <- "Predictor"
card_sum[24,3] <- "Response"

write.csv(card_sum, 'dataset_description.csv', row.names = F)
knitr::kable(card_sum, caption="Table 1. Credit Card Default Dataset")

## ----miss_plot, echo=FALSE, message=FALSE, warning=FALSE, eval = FALSE----
## 
## plot_missing(cc_data, title="Credit Card Data - Missing Values (%)")
## 

## ----descriptive statistics, echo=FALSE, message=FALSE, warning=FALSE, eval = FALSE----
## 
## #Use Describe Package to calculate Descriptive Statistic
## (CC_des <- describe(cc_data, na.rm=TRUE, interp=FALSE, skew=TRUE, ranges=TRUE, trim=.1, type=3, check=TRUE, fast=FALSE, quant=c(.1,.25,.75,.90), IQR=TRUE))
## 
## 

## ----echo=FALSE, message=FALSE, warning=FALSE----------------------------
#Temporary dataset turned to factors for visualization
cc_dataT <- cc_data
vars3 <- c("SEX", "MARRIAGE", "EDUCATION", "DEFAULT")

cc_dataT[vars3] <- lapply(cc_dataT[vars3], factor)

#Check ordering 
table(cc_dataT$PAY_0)

## ----bal, echo=FALSE, message=FALSE, warning=FALSE-----------------------
par(mfrow=(c(1,2)))
ggplot(cc_dataT, aes(x = LIMIT_BAL, fill = DEFAULT)) +
  geom_histogram() +
  labs(x = 'Credit Limit') +
  theme_gdocs()

## ----bal2, echo=FALSE, message=FALSE, warning=FALSE, eval=F--------------
## ggplot(cc_dataT, aes(x=LIMIT_BAL, y=LIMIT_BAL)) +
##   geom_boxplot()+
##   theme_pander()

## ----sex, echo=FALSE, message=FALSE, warning=FALSE, eval=F---------------
## ggplot(cc_dataT, aes(x = SEX, fill = DEFAULT)) +
##   geom_bar() +
##   labs(x = 'SEX') +
##   theme_pander()
## 

## ----edu, echo=FALSE, message=FALSE, warning=FALSE, eval=F---------------
## ggplot(cc_dataT, aes(x = EDUCATION, fill = DEFAULT)) +
##   geom_bar() +
##   labs(x = 'EDUCATION') +
##   theme_pander()
## 

## ----marriage, echo=FALSE, message=FALSE, warning=FALSE, eval=F----------
## ggplot(cc_dataT, aes(x = MARRIAGE, fill = DEFAULT)) +
##   geom_bar() +
##   labs(x = 'MARRIAGE') +
##   theme_pander()

## ----age, echo=FALSE, message=FALSE, warning=FALSE-----------------------
par(mfrow=(c(1,2)))
ggplot(cc_dataT, aes(x = AGE, fill = DEFAULT)) +
  geom_histogram() +
  labs(x = 'AGE') +
  theme_pander()

## ----age2, echo=FALSE, message=FALSE, warning=FALSE, eval=F--------------
## ggplot(cc_dataT, aes(x=AGE, y=AGE)) +
##   geom_boxplot()+
##   theme_pander()

## ----repay, echo=FALSE, message=FALSE, warning=FALSE---------------------

par(mfrow=(c(1,1)))
ggplot(stack(cc_dataT[,6:11]), aes(values, fill=ind))+
  facet_wrap(~ind, scales = "free") + 
  geom_bar() +
  theme_pander()+
  theme(legend.position="none")



## ----bill, echo=FALSE, message=FALSE, warning=FALSE----------------------

par(mfrow=(c(1,2)))
ggplot(stack(cc_dataT[,12:17]), aes(values, fill=ind))+
  facet_wrap(~ind, scales = "free") + 
  geom_histogram() +
  theme_pander() +
  theme(legend.position="none")

ggplot(stack(cc_dataT[,12:17]), aes(x = ind, y = values, fill=ind))+
  facet_wrap(~ind, scales = "free") + 
  geom_boxplot() +
  theme_pander() +
  theme(legend.position="none")

## ----pay, echo=FALSE, message=FALSE, warning=FALSE-----------------------

par(mfrow=(c(1,2)))
ggplot(stack(cc_dataT[,18:23]), aes(values, fill=ind))+
  facet_wrap(~ind, scales = "free") + 
  geom_histogram() +
  theme_pander() +
  theme(legend.position="none")

ggplot(stack(cc_dataT[,18:23]), aes(x = ind, y = values, fill=ind))+
  facet_wrap(~ind, scales = "free") + 
  geom_boxplot() +
  theme_pander() +
  theme(legend.position="none")

## ----default, echo=FALSE, message=FALSE, warning=FALSE-------------------
ggplot(cc_dataT, aes(x = DEFAULT)) +
  geom_bar(fill="blue") +
  labs(x = 'DEFAULT') +
  theme_pander()


## ----recode, echo=FALSE, message=FALSE, warning=FALSE--------------------
cc_data$MARRIED <- ifelse(cc_data$MARRIAGE==1, 1, 0)

cc_data$MALE <- ifelse(cc_data$SEX==1, 1, 0)

cc_data$EDU_COLLEGE <- ifelse(cc_data$EDUCATION %in% c(1, 2), 1, 0)
cc_data$EDU_ADV_DEGREE <- ifelse(cc_data$EDUCATION == 1, 1, 0) 

cc_dataR <- dplyr::select(cc_data, -EDUCATION, -MARRIAGE, -SEX)

## ----factorize, echo=FALSE, message=FALSE, warning=FALSE-----------------

# #Check for data type
sapply(cc_dataR,class)
# 
# #convert categorical to factor variables so that R can create dummy variables
# #ordered
# # vars1 <- c("EDUCATION", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
# # cc_data[vars1] <- lapply(cc_data[vars1], ordered)
# 
# #unordered
# vars2 <- c("MALE", "MARRIED","EDU_ADV_DEGREE", "EDU_COLLEGE","DEFAULT")
# cc_dataR[vars2] <- lapply(cc_dataR[vars2], factor)




## ----correlation, echo=FALSE, message=FALSE, warning=FALSE, fig.height=8, fig.width=10----
ggcorr(cc_dataR, method = "pairwise", label=TRUE, nbreaks=6)



## ----echo=FALSE, message=FALSE, warning=FALSE, fig.height=8, fig.width=10----
#all base variables
model1 <- glm(DEFAULT ~ .,
             data=cc_dataR,
             family = binomial(link="logit"))


#summary(model1)
car::vif(model1)

## ----pca, echo=FALSE, message=FALSE, warning=FALSE-----------------------
pca_val <- princomp(subset(cc_dataR, select = -DEFAULT))

#VB - 05/24/2018 added loading of variables
pca_df <- unclass(pca_val$loadings)
kable(pca_df)
screeplot(pca_val, type = "lines")

## ----echo=FALSE, message=FALSE, warning=FALSE, fig.height=8, fig.width=10----
cc_dataR <- cc_dataR %>% 
  mutate(AVG_BILL = rowMeans(cbind(BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6)), 
         AVG_BILL_TO_LIMIT = rowMeans(cbind(BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6))
         / LIMIT_BAL,
         PAY_TO_BILL = (rowMeans(cbind(PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6))+1)/
           (rowMeans(cbind(BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6))+1))

ggplot(cc_dataR, aes(x = AVG_BILL_TO_LIMIT)) +
  geom_histogram(shape=21, size = 4, 
     aes(fill = factor(DEFAULT))) + 
  labs(x = 'Average Bill to Limit') +
  theme_gdocs()



## ----echo=FALSE, message=FALSE, warning=FALSE----------------------------
cc_dataR$INC_COUNT = 0
for (i in 1:5) {
  cc_dataR$INC_COUNT <- ifelse(cc_dataR[[paste("BILL_AMT",i, sep = "")]] > cc_dataR[[paste("BILL_AMT",i+1, sep = "")]],
    cc_dataR$INC_COUNT +1, cc_dataR$INC_COUNT +0)
}

ggplot(cc_dataR, aes(x = INC_COUNT, fill = factor(DEFAULT))) +
  geom_bar() +
  labs(x = 'Increase in Bill Count') +
  theme_pander()


## ----dataprep, echo=FALSE, message=FALSE, warning=FALSE, fig.height=8, fig.width=10----

cc_dataN <- cc_dataR %>% 
  dplyr::select(-starts_with('BILL_AMT'))

# rename the response variable DEFAULT to target
cc_dataN$target <- factor(cc_dataR$DEFAULT)
cc_dataN$DEFAULT <- NULL

#colnames(cc_dataN)[15] <- "target"

#Remove rows with infinite values. Only 3 observations
cc_dataN <- cc_dataN[Reduce(`&`, lapply(cc_dataN, function(x) !is.na(x)  & is.finite(x))),]

#Normalize dataset
maxValue <- apply(cc_dataN, 2, max)  #CreditCardnn
minValue <- apply(cc_dataN, 2, min)   #CreditCardnn


smp <- floor(0.70 * nrow(cc_dataN))
set.seed(4784)
train_index <- sample(seq_len(nrow(cc_dataN)), size = smp, replace = FALSE)

# create training and test datasets 
train_all <- cc_dataN[train_index, ]
test_all <- cc_dataN[-train_index, ]

## Scale Test and Train for Neural Net 

train_nn <- train_all
train_nn$target <- as.integer(train_nn$target)

test_nn <- test_all
test_nn$target <- as.integer(test_nn$target)

#Normalize dataset
maxValueTrain <- apply(train_nn, 2, max)  #CreditCardnn
minValueTrain <- apply(train_nn, 2, min)   #CreditCardnn

maxValueTest <- apply(test_nn, 2, max)  #CreditCardnn
minValueTest <- apply(test_nn, 2, min)   #CreditCardnn

train_nn <- as.data.frame(scale(train_nn, center = minValueTrain, scale = maxValueTrain - minValueTrain))

test_nn <- as.data.frame(scale(test_nn, center = minValueTest, scale = maxValueTest - minValueTest))



## ----binary-logistic, echo=FALSE, message=FALSE, warning=FALSE-----------

model2 <- glm(target ~ .,
             data=cc_dataN,
             family = binomial(link="logit"))

#VIF measures are now acceptable
summary(model2)
car::vif(model2)


#Stepwise variable selection of model2
model2step <- step(model2, direction="both", trace=0)

summary(model2step)

#modified model metrics with stepwise variable selection

m2step_metrics <- calc_metrics("Model2 - STEP", model2step, test_all, train_all)
all_model_metrics <- rbind(all_model_metrics, m2step_metrics[[1]])

# capture the actual vs. predicted values
all_predictions[[1]] <- m2step_metrics[[3]]

# same for the confusion matrix, later used in the fourfoldplots in the metrics section
all_cm[[1]] <- m2step_metrics[[4]]


plot_varImp(model2)


## ----binary-logistic-out, echo=FALSE, message=FALSE, warning=FALSE-------

kable(m2step_metrics[[1]])


## ----ridge, echo=FALSE, message=FALSE, warning=FALSE---------------------

x <- as.matrix(train_all[,-23]) # Removes class
y <- as.double(as.matrix(train_all[, 23])) # Only class

# Fitting the model (Ridge: Alpha = 0)

cv.ridge <- cv.glmnet(x, y, family='binomial', alpha=0,  standardize=TRUE, type.measure='mse',nfolds=5)  


#create test data from test_all, removing the response variable
x_test <- as.matrix(test_all[,-23]) # Removes class
y_test <- as.double(as.matrix(test_all[, 23])) # Only class

#predict class, type="class"

m3ridge_metrics <- calc_metrics_2("Model3 - Ridge", cv.ridge, x_test, x, 
                                    cv.ridge$lambda.min, y_test)


all_model_metrics <- rbind(all_model_metrics, m3ridge_metrics[[1]])

# capture the actual vs. predicted values
all_predictions[[2]] <- m3ridge_metrics[[3]]

# same for the confusion matrix, later used in the fourfoldplots in the metrics section
all_cm[[2]] <- m3ridge_metrics[[4]]


#plot(fit.lasso, xvar="lambda")
#plot(fit10, main="LASSO")

#plot(cv.ridge$glmnet.fit, xvar="lambda")
#plot(fit0, main="Ridge")


## ----lasso, echo=FALSE, message=FALSE, warning=FALSE---------------------

cv.lasso <- cv.glmnet(x, y, family='binomial', alpha=1,  nfolds=5, standardize=TRUE, 
                       type.measure='mse' )  



#lasso_model <- cv.lasso$glmnet.fit
#summary(lasso_model)

#create test data from test_all, removing the response variable
x_test <- as.matrix(test_all[,-23]) # Removes class
y_test <- as.double(as.matrix(test_all[, 23])) # Only class

m4lasso_metrics <- calc_metrics_2("Model4 - Lasso", cv.lasso, x_test, x, 
                                    cv.lasso$lambda.1se, y_test)

all_model_metrics <- rbind(all_model_metrics, m4lasso_metrics[[1]])

# capture the actual vs. predicted values
all_predictions[[3]] <- m4lasso_metrics[[3]]

# same for the confusion matrix, later used in the fourfoldplots in the metrics section
all_cm[[3]] <- m4lasso_metrics[[4]]


## ------------------------------------------------------------------------

par(mfrow=c(1,2))

# Results
plot(cv.ridge)

# Results
plot(cv.lasso)


## ----echo=FALSE----------------------------------------------------------

ridge_coef <- as.data.frame(as.matrix(coef(cv.ridge, s=cv.ridge$lambda.min)))
ridge_coef$Variable <- rownames(ridge_coef); rownames(ridge_coef) <- NULL

lasso_coef <- as.data.frame(as.matrix(coef(cv.lasso, s=cv.lasso$lambda.1se)))
lasso_coef$Variable <- rownames(lasso_coef); rownames(lasso_coef) <- NULL

coef_all <- merge(ridge_coef,lasso_coef, by="Variable") 
coef_all <- coef_all %>% dplyr::select(Variable, Ridge=`1.x`, Lasso=`1.y`) %>% filter(Variable!="(Intercept)")

write.csv(coef_all, 'ridge_lasso_coef.csv', row.names = F)



## ----echo=FALSE----------------------------------------------------------
kable(coef_all, caption='Coefficient comparison between Ridge and Lasso')


## ----desicion-tree-------------------------------------------------------

m5 <- rpart(target ~ .,  data = train_all, method = 'class')
m5



## ------------------------------------------------------------------------
# additional plots
par(mfrow=c(1,2)) 
rsq.rpart(m5)

## ------------------------------------------------------------------------
plotcp(m5) # visualize cross-validation results


## ------------------------------------------------------------------------
summary(m5, cp = 0.1) # detailed summary of splits
#print(m5)

## ------------------------------------------------------------------------

# Mike - this plot throws an error "need finite ylim"



#plot(predict(m1), resid(m1))
#temp <- m1$frame[m1$frame$var == '<leaf>',]
#axis(3, at = temp$yval, as.character(row.names(temp)))
#mtext('leaf number', side = 3, line = 3)
#abline(h = 0, lty = 2)


## ------------------------------------------------------------------------
rpart.plot(m5, type=3, digits=3, fallen.leaves=TRUE, main='Regression Tree for Credit Card Defaults')


## ------------------------------------------------------------------------


m5_metrics <- calc_metrics_3("Model5 - DT", m5, test_all, train_all, "decision.tree")

all_model_metrics <- rbind(all_model_metrics, m5_metrics[[1]])

# capture the actual vs. predicted values
all_predictions[[4]] <- m5_metrics[[3]]

# same for the confusion matrix, later used in the fourfoldplots in the metrics section
all_cm[[4]] <- m5_metrics[[4]]



## ---- echo=FALSE, message=FALSE, warning=FALSE---------------------------

allVars <- colnames(cc_dataN)
predictorVars <- allVars[!allVars%in%'target']
predictorVars <- paste(predictorVars, collapse = "+")
(f <- as.formula(paste("target~", predictorVars, collapse = "+")))


## ---- echo=FALSE, message=FALSE, warning=FALSE---------------------------

neuralModel <- neuralnet(formula = f,linear.output = T, data = train_nn)


## ---- echo=FALSE, message=FALSE, warning=FALSE---------------------------

plot(neuralModel)


#kable(neuralModel$result.matrix)

## ---- echo=FALSE, message=FALSE, warning=FALSE---------------------------

# Sharon - sorry this changed when I switched to using the train_all dataset.  
# These variables were dropped as part of variable selection
# par(mfrow=c(2,2))
# gwplot(neuralModel, selected.covariate="SEX", min=-2.5, max=5)
# gwplot(neuralModel, selected.covariate="EDUCATION",
#               min=-2.5, max=5)
# gwplot(neuralModel, selected.covariate="MARRIAGE",
#               min=-2.5, max=5)
# gwplot(neuralModel, selected.covariate="AGE",
#              min=-2.5, max=5)

## ---- echo=FALSE, message=FALSE, warning=FALSE---------------------------

m6_metrics <- calc_metrics_3("Model6 - NNet", 
                             neuralModel, test_nn[, -23], train_nn, "nnet", test_nn[, 23])

all_model_metrics <- rbind(all_model_metrics, m6_metrics[[1]])

# capture the actual vs. predicted values
all_predictions[[5]] <- m6_metrics[[3]]

# same for the confusion matrix, later used in the fourfoldplots in the metrics section
all_cm[[5]] <- m6_metrics[[4]]


## ----fourfoldplots-------------------------------------------------------
par(mfrow=c(1,2))

fourfoldplot(m2step_metrics[[4]]$table, main="Logistic Regression")

fourfoldplot(m3ridge_metrics[[4]]$table, main="Ridge Regression")

fourfoldplot(m4lasso_metrics[[4]]$table, main="LASSO")

fourfoldplot(m5_metrics[[4]]$table, main="Decision Tree")

fourfoldplot(m6_metrics[[4]]$table, main="NeuralNet")


## ------------------------------------------------------------------------
perf_metrics1 <- dplyr::select(all_model_metrics,
                  Model,    Accuracy,   F1Score,    Kappa,  Sensitivity,    Specificity,    BalancedAccuracy)

perf_metrics2 <- all_model_metrics %>% transmute(Model, "Fale-Positive Rate" = 1- Specificity, Youden,
                                                 PosPredValue, NegPredValue)

# write the perf metrics out to csv to pull into the report as a table
write.csv(perf_metrics1, 'model_perf_metrics1.csv', row.names = F)

write.csv(perf_metrics2, 'model_perf_metrics2.csv', row.names = F)



