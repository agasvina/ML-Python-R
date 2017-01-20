dataset = read.csv("/Users/luca/git/ML-Python/data/Data.csv")
#To View the dataset: View(dataset)
#Let's clean this data:
#is.na checking if the value in the column is missing. 
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset $Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset $Salary)

#Encoding the catgorical data. 
#The vector function, we can do the same thing here
# c is vector in R
dataset$Country = factor(x = dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1,2,3)) #-> This won't do the one hot encoding for us though!
dataset$Purchased = factor(x = dataset$Purchased,
                         levels = c('No', 'Yes'),
                         labels = c(0,1))

#Splitting the data set: Train & Test 

#importing a helper library for us: caTools
#install.packages('caTools')
#To Select a library: library(caTools)
#We can change the seed:
set.seed(1783729)
split = sample.split(Y=dataset$Purchased,
                     SplitRatio = 0.8) #Spilt ratio will return true/false

#Creating Test & Training set:
training_set = subset(dataset, split== TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling:
training_set[,2:3] = scale(training_set[,2:3])
test_set[, 2:3] = scale(test_set[,2:3])
#The country and Purchase is not numeric yet, 
#So, we thrown an error: 'x' must be numeric
