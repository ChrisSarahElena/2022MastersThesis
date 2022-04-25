library(xlsx)

#Visualize dataset and get variable names
head(Final_Dataset_March_31)
names(Final_Dataset_March_31)

# Encoding Month_int with sine and cosine
Final_Dataset_March_31$Month_sin <- sin(2 * pi * Final_Dataset_March_31$Month_int/12)
Final_Dataset_March_31$Month_cos <- cos(2 * pi * Final_Dataset_March_31$Month_int/12)

# Verification: Plotting Cyclical Nature of Month
plot(month_sin, month_cos, col = "green", lwd = 3, main = "Cyclical Representation of Month Encoding")

#Export dataset
write.xlsx(Final_Dataset_March_31, file = "Disaster_encoded1.xlsx")


