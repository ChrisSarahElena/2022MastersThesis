library(readxl)
Test <- read_excel("Original_Dataset_No_Weather_TEST.xlsx")
Train <- read_excel("Original_Dataset_No_Weather_TRAIN.xlsx")
View(Test)
View(Train)

library(tidyverse)
library(caret)
library(dplyr)
library(leaps)
library(InformationValue)
library(kableExtra)

#Create new binary variable in both datasets
Test$cas <- ifelse(Test$Total == '0', 0, 1) 
Train$cas <- ifelse(Train$Total == '0', 0, 1) 

#Select the combination of variables
train.data <- Train %>%
  dplyr::select(1, 22, 25, 31, 39, 41, 43, 46, 58)
train.data

test.data <- Test %>%
  dplyr::select(1, 22, 25, 31, 39, 41, 43, 46, 58)
test.data

#Fit the model with the train set
model <- glm(cas ~., data = train.data, family = binomial)
summary(model)

#Calculate probabilities by making predictions, threshold 0.5
probabilities <- model %>% predict(test.data, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
head(probabilities)
head(predicted.classes)

#Check the results and some metrics with the confusion matrix, predictions vs actual classes
result <- caret::confusionMatrix(factor(predicted.classes), factor(test.data$cas), positive = "1")
result

metrics<-as.data.frame(result$byClass)
colnames(metrics)<-"metrics"
kable(round(metrics,4), caption = "F1-score, Precision and Recall ") %>%
  kable_styling(font_size = 16)

#Create new column containing the predictions
test.data$model_prob <- predict(model, test.data, type = "response")

#Convert the numeric columns to binary
test.data <- test.data  %>% mutate(probabilities = 1*(model_prob > .50) + 0)
test.data

test.data <- test.data %>% mutate(accurate = 1*(probabilities == cas))
sum(test.data$accurate)/nrow(test.data)

#Check AUC
plotROC(test.data$cas, test.data$model_prob)
