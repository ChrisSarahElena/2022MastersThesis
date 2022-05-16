                               #Croston's method#

# Required libraries
library(tsintermittent)


# # Croston Implementation for "x" from Original Dataset (Already split 80/20)

#Create object and sort containing oversampled training set
newdata_train1 <- Original_Dataset_No_Weather_TRAIN[order(Original_Dataset_No_Weather_TRAIN$Date),]

#Create and sort object containing test set
newdata_test1 <- Original_Dataset_No_Weather_TEST[order(Original_Dataset_No_Weather_TEST$Date),]

newtrain1 <- Original_Dataset_No_Weather_TRAIN$Total
newtest1 <- Original_Dataset_No_Weather_TEST$Total

new_croston1 <- crost(newtrain1,h=8401, type = "sbj", init = "mean", cost = "mar", outplot = F)

sum(newtest1) #sum of the test set values
sum(new_croston1$frc.out) # sum of the forecasted values


# Metrics
#library(Metrics)
#library(INDperform)

rmse(newtest1,new_croston1$frc.out)
nrmse(new_croston1$frc.out,newtest1)
mase(newtest1,new_croston1$frc.out)

##################################################
new_croston2 <- crost(newtrain1,h=8401, type = "sbj", init = "mean", cost = "msr", outplot = F)

sum(newtest1)
sum(new_croston2$frc.out)

# Metrics
#library(Metrics)

rmse(newtest1,new_croston2$frc.out)
nrmse(new_croston2$frc.out,newtest1)
mase(newtest1,new_croston2$frc.out)

##################################################
new_croston3 <- crost(newtrain1,h=8401, type = "sbj", init = "mean", cost = "mae", outplot = T)

sum(newtest1)
sum(new_croston3$frc.out)

# Metrics
#library(Metrics)

rmse(newtest1,new_croston3$frc.out)
nrmse(new_croston3$frc.out,newtest1)
mase(newtest1,new_croston3$frc.out)

##################################################
new_croston4 <- crost(newtrain1,h=8401, type = "sbj", init = "mean", cost = "mse", outplot = F)

sum(newtest1)
sum(new_croston4$frc.out)

# Metrics
#library(Metrics)

rmse(newtest1,new_croston4$frc.out)
nrmse(new_croston4$frc.out,newtest1)
mase(newtest1,new_croston4$frc.out)

##################################################
new_croston5 <- crost(newtrain1,h=8401, type = "sbj", init = "naive", cost = "mar", outplot = F)

sum(newtest1)
sum(new_croston5$frc.out)

# Metrics
#library(Metrics)

rmse(newtest1,new_croston5$frc.out)
nrmse(new_croston5$frc.out,newtest1)
mase(newtest1,new_croston5$frc.out)

##################################################
new_croston6 <- crost(newtrain1,h=8401, type = "sbj", init = "naive", cost = "msr", outplot = F)

sum(newtest1)
sum(new_croston6$frc.out)

# Metrics
#library(Metrics)

rmse(newtest1,new_croston6$frc.out)
nrmse(new_croston6$frc.out,newtest1)
mase(newtest1,new_croston6$frc.out)

##################################################
new_croston7 <- crost(newtrain1,h=8401, type = "sbj", init = "naive", cost = "mae", outplot = F)

sum(newtest1)
sum(new_croston7$frc.out)

# Metrics
#library(Metrics)

rmse(newtest1,new_croston7$frc.out)
nrmse(new_croston7$frc.out,newtest1)
mase(newtest1,new_croston7$frc.out)

##################################################
new_croston8 <- crost(newtrain1,h=8401, type = "sbj", init = "naive", cost = "mse", outplot = F)

sum(newtest1)
sum(new_croston8$frc.out)

# Metrics
#library(Metrics)
#library(INDperform)

rmse(newtest1,new_croston8$frc.out)
nrmse(new_croston8$frc.out,newtest1)
mase(newtest1,new_croston8$frc.out)


