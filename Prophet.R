
                            # Facebook Prophet #

# Required libraries
library(prophet)
library(writexl)


# Forecasting "Deceased" from original dataset #

# Creat object with prepared dataset
df2 <- Prophet_Deceased_old

#Run Prophet
m2 <- prophet(df2, seasonality.mode = 'additive',  mcmc.samples = 600)             

#Make dataframe with future dates for forecasting
future2 <- make_future_dataframe(m2, periods = 41, freq = 'month')
tail(future2)

# Generate prediction
fcst2 <- predict(m2, future2)

# Assign new column names
old_forecast_Deceased <- tail(fcst2[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])

#Plotting the forecast
plot(m2, fcst2, type = "l", ylabel = "\"Deceased\"", xlabel = "Year", main="Prophet: \"Total\" Variable from Original")                                                         

# Export the results
write_xlsx(as.data.frame(old_forecast_Deceased), " ... /Prophet_Old_Deceased_Forecast.xlsx")

#####################################################################################################

# Forecasting "Total" from original dataset #

df3 <- Prophet_Total_Old
m3 <- prophet(df3, seasonality.mode = 'additive',  mcmc.samples = 600)
future3 <- make_future_dataframe(m3, periods = 41, freq = 'month')
tail(future3)
fcst3 <- predict(m3, future3)
old_forecast_Total <- tail(fcst3[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])
plot(m3, fcst3, type = "l", ylabel = "\"Total\"", xlabel = "Year", main="Prophet: \"Total\" Variable from Original")

library("writexl")
write_xlsx(as.data.frame(old_forecast_Total), " ... /Prophet_Old_Total_Forecast.xlsx")


#####################################################################################################

# Forecasting "Deceased" from oversampled dataset #

df <- Oversampled_Prophet_Deceased
m <- prophet(df, seasonality.mode = 'additive', mcmc.samples = 600)
future <- make_future_dataframe(m, periods = 41, freq = 'month')
tail(future)
fcst <- predict(m, future)
new_forecast_Deceased <- tail(fcst[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])
plot(m, fcst, type = "l", ylabel = "\"Deceased\"", xlabel = "Year", main="Prophet: \"Deceased\" Outcome fom Oversampled")

library("writexl")
write_xlsx(as.data.frame(new_forecast_Deceased), "C: ... /Prophet_New_Deceased_Forecast.xlsx")

#####################################################################################################


# Forecasting "Total" from oversampled dataset #

df4 <- OG_Total_TRAIN
df4$cap <- 90000
m4 <- prophet(df4, seasonality.mode = 'additive', mcmc.samples = 600, growth = 'logistic')
future4 <- make_future_dataframe(m4, periods = 41, freq = 'month')
future4$cap <- 90000
tail(future4)
fcst4 <- predict(m4, future4)
new_forecast_Total <- tail(fcst4[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])
plot(m4, fcst4, type = "l", ylabel = "\"Total\"", xlabel = "Year", main="Prophet: \"Total\" Variable from Oversampled")

library("writexl")
write_xlsx(as.data.frame(fcst4), "C:/Users/Chris/OneDrive/Escritorio/KU Leuven/Master Thesis/Prophet/New version/fcst4_OG_TOTAL.xlsx")

plot_forecast_component(m4,fcst4)
plot(fcst4$yearly, type = "line")

#####################################################################################################



# Generate performance metrics
df.p <- performance_metrics(df.cv)
head(df.p)

# Plot desired metric
plot_cross_validation_metric(df.cv, metric = 'rmse') 


#Metrics and validation

#library(Metrics)
#library(INDperform)

rmse(Overall,Forecasted)
nrmse(Forecasted,Overall)
mase(Overall,Forecasted)
