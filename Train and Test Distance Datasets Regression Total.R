library(readxl)
Test <- read_excel("TestSet_0-3km.xlsx")
Train <- read_excel("Oversampled_TrainSet_0-3km.xlsx")
View(Test)
View(Train)

library(tidyverse)
library(caret)
library(dplyr)
library(leaps)
library(InformationValue)

Test$cas <- ifelse(Test$Total == '0', 0, 1) 
Train$cas <- ifelse(Train$Total == '0', 0, 1) 

#lapply(Test, function(x) {length(which(is.na(x))) })
#lapply(Train, function(x) {length(which(is.na(x))) })

train.data <- Train %>%
  dplyr::select(57, 61, 67, 69, 83, 92)
train.data

test.data <- Test %>%
  dplyr::select(56, 60, 66, 68, 82, 91)
test.data

test.data$cas <- as.factor(test.data$cas)
train.data$cas <- as.factor(train.data$cas)

model <- glm(cas ~., data = train.data, family = binomial)
summary(model)

probabilities <- model %>% predict(test.data, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "no", "yes")
head(probabilities)
head(predicted.classes)

table(test.data$cas, predicted.classes)

test.data$model_prob <- predict(model, test.data, type = "response")

test.data <- test.data  %>% mutate(probabilities = 1*(model_prob > .50) + 0)
test.data

test.data <- test.data %>% mutate(accurate = 1*(probabilities == cas))
sum(test.data$accurate)/nrow(test.data)

plotROC(test.data$cas, test.data$model_prob)
