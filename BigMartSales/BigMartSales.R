library(data.table) # used for reading and manipulation of data
library(dplyr) # used for data manipulation and joining
library(ggplot2) # used for ploting library(caret)      # used for modeling
library(caret) # used for modeling 
library(corrplot) # used for making correlation plot
library(xgboost) # used for building XGBoost model
library(cowplot)    # used for combining multiple plots 

train = fread("data/Train_UWu5bXk.csv") 
test = fread("data/Test_u94Q5KV.csv") 
submission = fread("data/SampleSubmission_TmnO39y.csv")

# train dataset has 8523 rows and 12 features and test has 5681 rows and 11 columns. train has 1 extra column which is the target variable. We will predict this target variable for the test dataset later in this tutorial.
dim(train);dim(test)

names(train)
names(test)

# In R, we have a pretty handy function called str(). It gives a short summary of all the features present in a dataframe. Let’s apply it on train and test data.
str(train)
str(test)

# So, we will go ahead combine both train and test data and will carry out data visualization, feature engineering, one-hot encoding, and label encoding. Later we would split this combined data back to train and test datasets.

test[,Item_Outlet_Sales := NA] 
combi = rbind(train, test) # combining train and test datasets dim(combi)
dim(combi)
# Let’s start with univariate EDA. It involves exploring variables individually. We will try to visualize the continuous variables using histograms and categorical variables using bar plots.
# Since our target variable is continuous, we can visualise it by plotting its histogram.
ggplot(train) + geom_histogram(aes(train$Item_Outlet_Sales), binwidth = 100, fill = "darkgreen") +  xlab("Item_Outlet_Sales")

# Now let’s check the numeric independent variables. We’ll again use the histograms for visualizations because that will help us in visualizing the distribution of the variables.
# There seems to be no clear-cut pattern in Item_Weight.
p1 = ggplot(combi) + geom_histogram(aes(Item_Weight), binwidth = 0.5, fill = "blue") 

# Item_Visibility is right-skewed and should be transformed to curb its skewness.
p2 = ggplot(combi) + geom_histogram(aes(Item_Visibility), binwidth = 0.005, fill = "blue") 

# We can clearly see 4 different distributions for Item_MRP. It is an interesting insight.
p3 = ggplot(combi) + geom_histogram(aes(Item_MRP), binwidth = 1, fill = "blue") 
plot_grid(p1, p2, p3, nrow = 1) # plot_grid() from cowplot package

# Now we’ll try to explore and gain some insights from the categorical variables. A categorical variable or feature can have only a finite set of values. Let’s first plot Item_Fat_Content.
ggplot(combi %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) +   geom_bar(aes(Item_Fat_Content, Count), stat = "identity", fill = "coral1")

# In the figure above, ‘LF’, ‘low fat’, and ‘Low Fat’ are the same category and can be combined into one. Similarly we can be done for ‘reg’ and ‘Regular’ into one. After making these corrections we’ll plot the same figure again.
combi$Item_Fat_Content[combi$Item_Fat_Content == "LF"] = "Low Fat" 
combi$Item_Fat_Content[combi$Item_Fat_Content == "low fat"] = "Low Fat" 
combi$Item_Fat_Content[combi$Item_Fat_Content == "reg"] = "Regular" 
ggplot(combi %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) +   geom_bar(aes(Item_Fat_Content, Count), stat = "identity", fill = "coral1")

# Now let’s check the other categorical variables.
# plot for Item_Type 
p4 = ggplot(combi %>% group_by(Item_Type) %>% summarise(Count = n())) +   geom_bar(aes(Item_Type, Count), stat = "identity", fill = "coral1") +  xlab("") +  geom_label(aes(Item_Type, Count, label = Count), vjust = 0.5) +  theme(axis.text.x = element_text(angle = 45, hjust = 1))+  ggtitle("Item_Type")

# plot for Outlet_Identifier 
p5 = ggplot(combi %>% group_by(Outlet_Identifier) %>% summarise(Count = n())) +   geom_bar(aes(Outlet_Identifier, Count), stat = "identity", fill = "coral1") +  geom_label(aes(Outlet_Identifier, Count, label = Count), vjust = 0.5) +  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# plot for Outlet_Size 
# In Outlet_Size’s plot, for 4016 observations, Outlet_Size is blank or missing.We will check for this in the bivariate analysis to substitute the missing values in the Outlet_Size.
p6 = ggplot(combi %>% group_by(Outlet_Size) %>% summarise(Count = n())) +   geom_bar(aes(Outlet_Size, Count), stat = "identity", fill = "coral1") +  geom_label(aes(Outlet_Size, Count, label = Count), vjust = 0.5) +  theme(axis.text.x = element_text(angle = 45, hjust = 1))
second_row = plot_grid(p5, p6, nrow = 1) 
plot_grid(p4, second_row, ncol = 1)

# We’ll also check the remaining categorical variables.

# plot for Outlet_Establishment_Year 
p7 = ggplot(combi %>% group_by(Outlet_Establishment_Year) %>% summarise(Count = n())) +   geom_bar(aes(factor(Outlet_Establishment_Year), Count), stat = "identity", fill = "coral1") +  geom_label(aes(factor(Outlet_Establishment_Year), Count, label = Count), vjust = 0.5) +  xlab("Outlet_Establishment_Year") +  theme(axis.text.x = element_text(size = 8.5))

# plot for Outlet_Type 
p8 = ggplot(combi %>% group_by(Outlet_Type) %>% summarise(Count = n())) +   geom_bar(aes(Outlet_Type, Count), stat = "identity", fill = "coral1") +  geom_label(aes(factor(Outlet_Type), Count, label = Count), vjust = 0.5) +  theme(axis.text.x = element_text(size = 8.5))

# ploting both plots together 
plot_grid(p7, p8, nrow = 1)

############# Bivariate Analysis ##############
train = combi[1:nrow(train)] # extracting train data from the combined data

# Item_Weight vs Item_Outlet_Sales 
p9 = ggplot(train) +      geom_point(aes(Item_Weight, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +     theme(axis.title = element_text(size = 8.5))

# Item_Visibility vs Item_Outlet_Sales 
p10 = ggplot(train) +       geom_point(aes(Item_Visibility, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +      theme(axis.title = element_text(size = 8.5))

# Item_MRP vs Item_Outlet_Sales 
p11 = ggplot(train) +       geom_point(aes(Item_MRP, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +      theme(axis.title = element_text(size = 8.5))

second_row_2 = plot_grid(p10,p11, ncol=2)
plot_grid(p9,second_row_2, ncol=1)
# In the third plot of Item_MRP vs Item_Outlet_Sales, we can clearly see 4 segments of prices that can be used in feature engineering to create a new variable.


# Item_Type vs Item_Outlet_Sales 
p12 = ggplot(train) +       geom_violin(aes(Item_Type, Item_Outlet_Sales), fill = "magenta") +      theme(axis.text.x = element_text(angle = 45, hjust = 1),            axis.text = element_text(size = 6),            axis.title = element_text(size = 8.5))

# Item_Fat_Content vs Item_Outlet_Sales 
p13 = ggplot(train) +       geom_violin(aes(Item_Fat_Content, Item_Outlet_Sales), fill = "magenta") +      theme(axis.text.x = element_text(angle = 45, hjust = 1),            axis.text = element_text(size = 8),            axis.title = element_text(size = 8.5))

# Outlet_Identifier vs Item_Outlet_Sales 
p14 = ggplot(train) +       geom_violin(aes(Outlet_Identifier, Item_Outlet_Sales), fill = "magenta") +      theme(axis.text.x = element_text(angle = 45, hjust = 1),            axis.text = element_text(size = 8),            axis.title = element_text(size = 8.5))

second_row_3 = plot_grid(p13, p14, ncol = 2) 
plot_grid(p12, second_row_3, ncol = 1)
# The distribution for OUT010 and OUT019 categories of Outlet_Identifier are quite similar and very much different from the rest of the categories of Outlet_Identifier.

# In the univariate analysis, we came to know about the empty values in Outlet_Size variable. Let’s check the distribution of the target variable across Outlet_Size.
ggplot(train) + geom_violin(aes(Outlet_Size, Item_Outlet_Sales), fill = "magenta")
# The distribution of ‘Small’ Outlet_Size is almost identical to the distribution of the blank category (first vioin) of Outlet_Size. So, we can substitute the blanks in Outlet_Size with ‘Small’.
#Please note that this is not the only way to impute missing values, but for the time being we will go ahead and impute the missing values with ‘Small’.

# Let’s examine the remaining variables.
p15 = ggplot(train) + geom_violin(aes(Outlet_Location_Type, Item_Outlet_Sales), fill = "magenta") 
p16 = ggplot(train) + geom_violin(aes(Outlet_Type, Item_Outlet_Sales), fill = "magenta") 
plot_grid(p15, p16, ncol = 1)
# Tier 1 and Tier 3 locations of Outlet_Location_Type look similar.
# In the Outlet_Type plot, Grocery Store has most of its data points around the lower sales values as compared to the other categories.

############################# Missing Values Treatments #######################
# You can try the following code to quickly find missing values in a variable.
sum(is.na(combi$Item_Weight))

# We’ll now impute Item_Weight with mean weight based on the Item_Identifier variable.
missing_index = which(is.na(combi$Item_Weight)) 
for(i in missing_index){    
  item = combi$Item_Identifier[i]  
  combi$Item_Weight[i] = mean(combi$Item_Weight[combi$Item_Identifier == item], na.rm = T) 
}

sum(is.na(combi$Item_Weight))

ggplot(combi) + geom_histogram(aes(Item_Visibility), bins = 100)

zero_index = which(combi$Item_Visibility == 0) 
for(i in zero_index){    
  item = combi$Item_Identifier[i]  
  combi$Item_Visibility[i] = mean(combi$Item_Visibility[combi$Item_Identifier == item], na.rm = T)  
}

ggplot(combi) + geom_histogram(aes(Item_Visibility), bins = 100)


################# Feature Engineering ##############################3
# In this section we will create the following new features:
  
# Item_Type_new: Broader categories for the variable Item_Type.
# Item_category: Categorical variable derived from Item_Identifier.
# Outlet_Years: Years of operation for outlets.
# price_per_unit_wt: Item_MRP/Item_Weight
# Item_MRP_clusters: Binned feature for Item_MRP.

print(combi$Item_Type)

# We can have a look at the Item_Type variable and classify the categories into perishable and non_perishable as per our understanding and make it into a new feature.

perishable = c("Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood")
non_perishable = c("Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Health and Hygiene", "Household", "Soft Drinks")

# create a new feature 'Item_Type_new' 
combi[,Item_Type_new := ifelse(Item_Type %in% perishable, "perishable", ifelse(Item_Type %in% non_perishable, "non_perishable", "not_sure"))]

# Let’s compare Item_Type with the first 2 characters of Item_Identifier, i.e., ‘DR’, ‘FD’, and ‘NC’. These identifiers most probably stand for drinks, food, and non-consumable.

table(combi$Item_Type, substr(combi$Item_Identifier, 1, 2))
# table(combi$Item_Type)

# Based on the above table we can create a new feature. Let’s call it Item_category.
combi[,Item_category := substr(combi$Item_Identifier, 1, 2)]
# table(combi$Item_Identifier)

# We will also change the values of Item_Fat_Content wherever Item_category is ‘NC’ because non-consumable items cannot have any fat content. We will also create a couple of more features — Outlet_Years (years of operation) and price_per_unit_wt (price per unit weight).
combi$Item_Fat_Content[combi$Item_category == "NC"] = "Non-Edible" 
# years of operation for outlets 
combi[,Outlet_Years := 2013 - Outlet_Establishment_Year] 
combi$Outlet_Establishment_Year = as.factor(combi$Outlet_Establishment_Year)## what is this for?

# Price per unit weight 
combi[,price_per_unit_wt := Item_MRP/Item_Weight]

# Earlier in the Item_MRP vs Item_Outlet_Sales plot, we saw Item_MRP was spread across in 4 chunks. Now let’s assign a label to each of these chunks and use this label as a new variable.

# creating new independent variable - Item_MRP_clusters 
combi[,Item_MRP_clusters := ifelse(Item_MRP < 69, "1st", ifelse(Item_MRP >= 69 & Item_MRP < 136, "2nd", ifelse(Item_MRP >= 136 & Item_MRP < 203, "3rd", "4th")))]

################### Encoding Categorical Variables ###########
# We will label encode Outlet_Size and Outlet_Location_Type as these are ordinal variables.
combi[,Outlet_Size_num := ifelse(Outlet_Size == "Small", 0,                                 ifelse(Outlet_Size == "Medium", 1, 2))] 

combi[,Outlet_Location_Type_num := ifelse(Outlet_Location_Type == "Tier 3", 0,                                          ifelse(Outlet_Location_Type == "Tier 2", 1, 2))] 

# removing categorical variables after label encoding 
combi[, c("Outlet_Size", "Outlet_Location_Type") := NULL]

# One hot encoding for the categorical variable
ohe = dummyVars("~.", data = combi[,-c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")], fullRank = T) 
ohe_df = data.table(predict(ohe, combi[,-c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")])) 
combi = cbind(combi[,"Item_Identifier"], ohe_df)

##################### Pre processing of Data ###############3333
# Some machine learning methods assume normally distributed data and a skewed variable can be transformed by taking its log, square root, or cube root so as to make its distribution as close to normal distribution as possible. In our data, variables Item_Visibility and price_per_unit_wt are highly skewed. So, we will treat their skewness with the help of log transformation.
# combi[,Item_Visibility]
combi[,Item_Visibility := log(Item_Visibility + 1)] # log + 1 to avoid division by zero 

# combi[,price_per_unit_wt]
combi[,price_per_unit_wt := log(price_per_unit_wt + 1)]
# combi[,price_per_unit_wt]
# combi[,Item_Visibility]

# Let’s scale and center the numeric variables to make them have a mean of zero, standard deviation of one and scale of 0 to 1. Scaling and centering is required for linear regression models.
num_vars = which(sapply(combi, is.numeric)) # index of numeric features 
num_vars_names = names(num_vars) 
combi_numeric = combi[,setdiff(num_vars_names, "Item_Outlet_Sales"), with = F] 
prep_num = preProcess(combi_numeric, method=c("center", "scale")) 
combi_numeric_norm = predict(prep_num, combi_numeric)
combi[,setdiff(num_vars_names, "Item_Outlet_Sales") := NULL] # removing numeric independent variables 
combi = cbind(combi, combi_numeric_norm)

# Splitting the combined data combi back to train and test set.
train = combi[1:nrow(train)] 
test = combi[(nrow(train) + 1):nrow(combi)] 
test[,Item_Outlet_Sales := NULL] # removing Item_Outlet_Sales as it contains only NA for test dataset

# It is not desirable to have correlated features if we are using linear regressions.
cor_train = cor(train[,-c("Item_Identifier")]) 
corrplot(cor_train, method = "pie", type = "lower", tl.cex = 0.9)



######################## Linear Regression ########################
# Building Model
linear_reg_mod = lm(Item_Outlet_Sales ~ ., data = train[,-c("Item_Identifier")])

# Making Predictions on test Data
# preparing dataframe for submission and writing it in a csv file 
submission$Item_Outlet_Sales = predict(linear_reg_mod, test[,-c("Item_Identifier")]) 
write.csv(submission, "Linear_Reg_submit.csv", row.names = F)

######################## Regularized Linear Regression ########################
# Regularised regression models can handle the correlated independent variables well and helps in overcoming overfitting.
# Ridge penalty shrinks the coefficients of correlated predictors towards each other, 
# while the Lasso tends to pick one of a pair of correlated features and discard the other.
# The tuning parameter lambda controls the strength of the penalty.
# https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/

# Lasso Regression
set.seed(1235)
my_control = trainControl(method="cv", number=5) 
Grid = expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0002)) 
lasso_linear_reg_mod = train(x = train[, -c("Item_Identifier", "Item_Outlet_Sales")], y = train$Item_Outlet_Sales, method='glmnet', trControl= my_control, tuneGrid = Grid)
# Mean validation score: 1130.02
# Leaderboard score: 1202.26

# Ridge Regression
set.seed(1236) 
my_control = trainControl(method="cv", number=5) 
Grid = expand.grid(alpha = 0, lambda = seq(0.001,0.1,by = 0.0002)) 
ridge_linear_reg_mod = train(x = train[, -c("Item_Identifier", "Item_Outlet_Sales")], y = train$Item_Outlet_Sales,method='glmnet', trControl= my_control, tuneGrid = Grid)
# Mean validation score: 1135.08
# Leaderboard score: 1219.08

################################### Random Forest #####################################################
# RandomForest is a tree based bootstrapping algorithm wherein a certain number of weak learners
# (decision trees) are combined to make a powerful prediction model. 
# For every individual learner, a random sample of rows and a few randomly chosen variables are 
# used to build a decision tree model. Final prediction can be a function of all the predictions
# made by the individual learners. In case of a regression problem, the final prediction can be
# mean of all the predictions.
# https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/

# We will now build a RandomForest model with 400 trees. 
# The other tuning parameters used here are mtry — no. of predictor variables randomly sampled at each 
# split, and min.node.size — minimum size of terminal nodes (setting this number large causes smaller 
# trees and reduces overfitting).

set.seed(1237)
my_control = trainControl(method="cv", number=5) # 5-fold CV
tgrid = expand.grid(
  .mtry = c(3:10),
  .splitrule = "variance",
  .min.node.size = c(10,15,20)
)

rf_mod = train(x = train[, -c("Item_Identifier", "Item_Outlet_Sales")],
               y = train$Item_Outlet_Sales,
               method='ranger',
               trControl= my_control,
               tuneGrid = tgrid,
               num.trees = 400,
               importance = "permutation")

# Mean validation score: 1088.05
# Leaderboard score: 1157.25

# Our score on the leaderboard has improved considerably by using RandomForest. 
# Now let’s visualize the RMSE scores for different tuning parameters.
plot(rf_mod)

# As per the plot shown above, the best score is achieved at mtry = 5 and min.node.size = 20.

# Let’s plot feature importance based on the RandomForest model
plot(varImp(rf_mod))
# As expected Item_MRP is the most important variable in predicting the target variable. New features created by us, like price_per_unit_wt, Outlet_Years, Item_MRP_Clusters, are also among the top most important variables. 
# This is why feature engineering plays such a crucial role in predictive modeling.

##########################################################################