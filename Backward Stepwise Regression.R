library(readxl)
Final <- read_excel("Normalized Dataset.xlsx")
View(Final)

library(tidyverse)
library(caret)
library(dplyr)
library(leaps)
library(statsr)
  
#Extract the columns that are linearly related
rankifremoved <- sapply(1:ncol(Final), function (x) qr(Final[,-x])$rank)
which(rankifremoved == max(rankifremoved))

finals <- Final %>%
  dplyr::select(10, 56:58, 60, 62, 66:68, 72, 82)
finals

library(MASS)
#Fit the full model 
full.model <- lm(Total ~., data = finals)
#Stepwise regression model
step.model <- stepAIC(full.model, direction = "both", trace = FALSE)
summary(step.model)

models <- regsubsets(Total ~., data = finals, nvmax = 5,
                     method = "seqrep")
summary(models)

set.seed(123)
#Set up repeated k-fold cross-validation
train.control <- trainControl(method = "cv", number = 10)
#Train the model
step.model <- train(Total ~., data = finals,
                    method = "leapBackward", 
                    tuneGrid = data.frame(nvmax = 1:5),
                    trControl = train.control
)

step.model$results

step.model$bestTune

summary(step.model$finalModel)
