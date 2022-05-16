# import the library
library(caret)

# dataset
data <- Final_datase_na_omit_onlyNumeric
data$Date <- as.numeric(data$Date)

# custom function to implement min max scaling
minMax <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

#normalise data using custom function
normaliseddata <- as.data.frame(lapply(data, minMax))
head(normaliseddata)


#Split
smp_size <- floor(0.80 * nrow(normaliseddata))

set.seed(123)

train <- na.omit(normaliseddata[1:35563, ]) #Train set
test <- na.omit(normaliseddata[35564:44454, ]) #Test set



#Export
library("writexl")
write_xlsx(as.data.frame(normaliseddata), "C:/Users/Chris/OneDrive/Escritorio/KU Leuven/Master Thesis/Norm_Max-Min.xlsx")

write_xlsx(as.data.frame(train), "C:/Users/Chris/OneDrive/Escritorio/KU Leuven/Master Thesis/Norm_Max-Min_TRAIN.xlsx")

write_xlsx(as.data.frame(test), "C:/Users/Chris/OneDrive/Escritorio/KU Leuven/Master Thesis/Norm_Max-Min_TEST.xlsx")

View(test)
