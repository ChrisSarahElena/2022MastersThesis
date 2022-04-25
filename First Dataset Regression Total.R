library(readxl)
Second <- read_excel("Initial Dataset.xlsx")
View(Second)

library(tidyverse)
library(caret)
library(dplyr)
library(leaps)

Second$cas <- ifelse(Second$Total == '0', 0, 1) 

#Check for missing values
lapply(Second, function(x) {length(which(is.na(x))) })

subsec <- Second %>%
  na.omit() %>%
  dplyr::select(15, 25, 32, 39:40, 45, 55)
subsec

#Apply understandable format
subsec <- as.data.frame(unclass(subsec), stringsAsFactors = TRUE)

#Find the variables with the best p-values
subsec %>% 
  pivot_longer(-cas) %>% 
  group_split(name) %>% 
  set_names(nm = map(., ~ first(.x$name))) %>% 
  map(~ tidy(lm(cas ~ value, data = .))) %>% 
  map(~ filter(., p.value < 0.05))

#Fit the model
set.seed(123)
training.samples <- subsec$cas %>% 
#Create partitions
createDataPartition(p = 0.8, list = FALSE)
train.data  <- subsec[training.samples, ]
test.data <- subsec[-training.samples, ]

#Train the model
model <- glm(cas ~., data = train.data, family = binomial)
summary(model)

#Make predictions
probabilities <- model %>% predict(test.data, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "no", "yes")
head(probabilities)

test.data$model_prob <- predict(model, test.data, type = "response")

test.data <- test.data  %>% mutate(probabilities = 1*(model_prob > .50) + 0)
test.data

#Evaluation
test.data <- test.data %>% mutate(accurate = 1*(probabilities == cas))
sum(test.data$accurate)/nrow(test.data)

library(InformationValue)
misClassError(test.data$cas, test.data$model_prob, threshold = 0.5)
