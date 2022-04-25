                               #Croston's method#

# Required libraries
library(tsintermittent)


# # Croston Implementation for "Deceased" from Original Dataset

# Sort dataset by date
data1 <- Disaster_Clean4[order(Final_Dataset_April_12_Time$Date),]

# Split 80/20 train set
train1 <- data1$Deceased[1:36052]
test1 <- data1$Deceased[36053:36221] # 169 observations from the test set

# Apply Croston | Modify hyperparameters accordingly
croston1 <- crost(train1,h=169, type = "sbj", init = "mean", cost = "mse", outplot = F)

#Validate results
sum(test1)          #Sum of the test set
sum(croston1$frc.out) #Sum of the generated values forecasted for horizon

##########################################################################################
#
# # Croston Implementation for "Total" from Original Dataset

# Sort dataset by date
data2 <- Disaster_Clean4[order(Final_Dataset_April_12_Time$Date),]

# Split 80/20 train set
train2 <- data2$Total[1:36052]
test2 <- data2$Total[36053:36221] # 169 observations from the test set

# Apply Croston | Modify hyperparameters accordingly
croston2 <- crost(train2,h=169, type = "sbj", init = "mean", cost = "mse", outplot = F)

#Validate results
sum(test2)          #Sum of the test set
sum(croston2$frc.out) #Sum of the generated values forecasted for horizon


##########################################################################################
#
# # Croston Implementation for "Deceased" from Oversampled Dataset (Already split 80/20)

#Create object and sort containing oversampled training set
newdata_train1 <- Oversampled_dataset_Final[order(Oversampled_dataset_Final$Date),]

#Create and sort object containing normalized test set
newdata_test1 <- normalized_dataset_sqrt_TestSet[order(normalized_dataset_sqrt_TestSet$Date),]

newtrain1 <- newdata_train1$Deceased
newtest1 <- newdata_test1$Deceased[1:169]

new_croston1 <- crost(newtrain1,h=169, type = "sbj", init = "naive", cost = "mae", outplot = F)

sum(newtest1)
sum(new_croston1$frc.out)

##########################################################################################
#
# # Croston Implementation for "Total" from Oversampled Dataset (Already split 80/20)

#Create object and sort containing oversampled training set
newdata_train2 <- Oversampled_dataset_Final[order(Oversampled_dataset_Final$Date),]

#Create and sort object containing normalized test set
newdata_test2 <- normalized_dataset_sqrt_TestSet[order(normalized_dataset_sqrt_TestSet$Date),]

newtrain2 <- newdata_train2$Total
newtest2 <- newdata_test2$Total[1:169]

new_croston2 <- crost(newtrain2,h=169, type = "sbj", init = "naive", cost = "mae", outplot = F)

sum(newtest2)
sum(new_croston2$frc.out)

##############################################################################################