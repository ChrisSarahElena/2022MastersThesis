library(readxl)
Final <- read_excel("Oversampled Overall Dataset")
View(Final)

library(tidyverse)
library(caret)
library(dplyr)
library(broom)

Final$cas <- ifelse(Final$Total == '0', 0, 1) 

finals <- Final %>%
  select(56, 60, 66, 68, 82, 91)
finals

finals <- as.data.frame(unclass(finals), stringsAsFactors = TRUE)

finals %>% 
  pivot_longer(-cas) %>% 
  group_split(name) %>% 
  set_names(nm = map(., ~ first(.x$name))) %>% 
  map(~ tidy(lm(cas ~ value, data = .))) %>% 
  map(~ filter(., p.value < 0.05))


set.seed(123)
training.samples <- finals$cas %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- finals[training.samples, ]
test.data <- finals[-training.samples, ]

model <- glm(cas ~., data = train.data, family = binomial)
summary(model)

probabilities <- model %>% predict(test.data, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "no", "yes")
head(probabilities)

test.data$model_prob <- predict(model, test.data, type = "response")

test.data <- test.data  %>% mutate(probabilities = 1*(model_prob > .50) + 0)
test.data

test.data <- test.data %>% mutate(accurate = 1*(probabilities == cas))
sum(test.data$accurate)/nrow(test.data)







