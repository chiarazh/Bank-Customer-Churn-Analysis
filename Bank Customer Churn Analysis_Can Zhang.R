# Bank Customer Churn Analysis
# HarvardX Data Science: Capstone 2 (Choose Your Own Project)
# Can Zhang


####Install and load packages####

if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(tidymodels)) install.packages("tidymodels", repos = "http://cran.us.r-project.org")
if(!require(MLmetrics)) install.packages("MLmetrics", repos = "http://cran.us.r-project.org")
if(!require(lightgbm)) install.packages("lightgbm", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")


library(readr)
library(tidyverse)
library(dplyr)
library(caret)
library(ggplot2)
library(gridExtra)
library(ranger)
library(tidymodels)
library(MLmetrics)
library(lightgbm)
library(knitr)

# Check the current working directory

getwd()

####Download and create data set####

# Download file
# Please note that the original data set is downloaded from Kaggle: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
# To facilitate the process, I uploaded the data set to my github

download.file("https://raw.githubusercontent.com/chiarazh/Bank-Customer-Churn-Analysis/main/Churn_Modelling.csv", "Churn_Modelling.csv")

# Read data from csv format

bank <- read_csv("Churn_Modelling.csv" )

# Check basic info of the data set

dim(bank)
head(bank)
str(bank, give.attr = FALSE)

# Drop irrelevant variables "RowNumber", "CustomerId" and "Surname"

bank <- bank %>% select(-c("RowNumber", "Surname", "CustomerId"))
head(bank)

# Check if there are NA values

na_values <- cbind(lapply(lapply(bank, is.na), sum))
na_values

# Change "Geography", "Gender","Exited", "IsActiveMember" and "HasCrCard" from numeric to factor

bank_temp <- bank

bank_temp <- bank_temp %>% mutate(Exited = ifelse(Exited == "1", "yes", "no") %>% 
                                    factor(levels = c("yes", "no"))) ## Exited(0)=no, Exited(1)=yes

bank_temp$Geography <- as.factor(bank_temp$Geography)
bank_temp$Gender <- as.factor(bank_temp$Gender)
bank_temp$IsActiveMember <- as.factor(bank_temp$IsActiveMember)
bank_temp$HasCrCard <- as.factor(bank_temp$HasCrCard)
str(bank_temp)



####Generate train set(bank1) and validation set(final hold-out set)####
#Validation set will be 10% of the bank data
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = bank_temp$Exited, times = 1, p = 0.1, list = FALSE)
validation <- bank_temp[test_index,]
bank1 <- bank_temp[-test_index,]

dim(bank1)
table(bank1$Exited)

dim(validation)
table(validation$Exited)

####Data Exploration####

dim(bank1)
head(bank1)
str(bank1)
table(bank1$Exited)

churn_rate <- sum(bank1$Exited == "yes")/nrow(bank1)
churn_rate

#####Churn and geography#####
churn_country <- bank1 %>% group_by(Geography) %>% 
  summarise(total_customer = n(),
            churn = sum(Exited == "yes"),
            stay = total_customer - churn,
            churn_rate = churn/total_customer)
churn_country 

plot_country <- churn_country %>% select(Geography, churn, stay) %>% 
  pivot_longer(!Geography, names_to = "status", values_to = "count") %>% 
  ggplot(aes(x=Geography, y = count, fill = status)) +
  geom_bar(position="stack", stat="identity") +
  ggtitle("Churn info by geography") +
  theme_minimal()
plot_country

plot_country_rate <- churn_country %>% ggplot(aes(Geography, churn_rate)) +
  geom_point() +
  geom_hline(yintercept = churn_rate, color = "red") +
  ggtitle("Churn rate by geography") +
  theme_minimal()
plot_country_rate

#####Churn and gender####
churn_gender <- bank1 %>% 
  group_by(Gender) %>% 
  summarise(total_customer = n(),
            churn = sum(Exited== "yes"),
            stay = total_customer - churn,
            churn_rate = churn/total_customer)
churn_gender

plot_gender <- churn_gender %>% 
  select(Gender, churn, stay) %>% 
  pivot_longer(!Gender, names_to = "status", values_to = "count") %>% 
  ggplot(aes(x = Gender, y = count, fill = status)) +
  geom_bar(position = "stack", stat = "identity") +
  ggtitle("Churn info by gender") +
  theme_minimal()
plot_gender

plot_gender_rate <- churn_gender %>% ggplot(aes(Gender, churn_rate)) +
  geom_point() +
  geom_hline(yintercept = churn_rate, color = "red") +
  ggtitle("Churn rate by gender") +
  theme_minimal()
plot_gender_rate 

#####Churn and NumOfProducts#####
churn_product <- bank1 %>% 
  group_by(NumOfProducts) %>% 
  summarise(total_customer = n(),
            churn = sum(Exited == "yes"),
            stay = total_customer - churn,
            churn_rate = churn/total_customer)
churn_product

plot_product <- churn_product %>% 
  select(NumOfProducts, churn, stay) %>% 
  pivot_longer(!NumOfProducts, names_to = "status", values_to = "count") %>% 
  ggplot(aes(x = NumOfProducts, y = count, fill = status)) +
  geom_bar(position = "stack", stat = "identity") +
  ggtitle("Churn info by number of products") +
  theme_minimal()
plot_product 

plot_product_rate <- churn_product %>% ggplot(aes(NumOfProducts, churn_rate)) +
  geom_point() +
  geom_hline(yintercept = churn_rate, color = "red") +
  ggtitle("Churn rate by gender") +
  theme_minimal()
plot_product_rate

#####Churn and HasCrCard#####
churn_crcard <- bank1 %>% 
  group_by(HasCrCard) %>% 
  summarise(total_customer = n(),
            churn = sum(Exited == "yes"),
            stay = total_customer - churn,
            churn_rate = churn/total_customer)
churn_crcard

plot_crcard <- churn_crcard %>% 
  select(HasCrCard, churn, stay) %>% 
  pivot_longer(!HasCrCard, names_to = "status", values_to = "count") %>% 
  ggplot(aes(x = HasCrCard, y = count, fill = status)) +
  geom_bar(position = "stack", stat = "identity") +
  ggtitle("Churn info by credit card possession") +
  theme_minimal()
plot_crcard 

plot_crcard_rate <- churn_crcard %>% ggplot(aes(HasCrCard, churn_rate)) +
  geom_point() +
  geom_hline(yintercept = churn_rate, color = "red") +
  ggtitle("Churn rate by credit card possession") +
  theme_minimal()
plot_crcard_rate 

#####Churn and IsActiveMember#####
churn_active <- bank1 %>% 
  group_by(IsActiveMember) %>% 
  summarise(total_customer = n(),
            churn = sum(Exited == "yes"),
            stay = total_customer - churn,
            churn_rate = churn/total_customer)
churn_active

plot_active <- churn_active %>% 
  select(IsActiveMember, churn, stay) %>% 
  pivot_longer(!IsActiveMember, names_to = "status", values_to = "count") %>% 
  ggplot(aes(x = IsActiveMember, y = count, fill = status)) +
  geom_bar(position = "stack", stat = "identity") +
  ggtitle("Churn info by activeness") +
  theme_minimal()
plot_active

plot_active_rate <- churn_active %>% ggplot(aes(IsActiveMember, churn_rate)) +
  geom_point() +
  geom_hline(yintercept = churn_rate, color = "red") +
  ggtitle("Churn rate by activeness") +
  theme_minimal()
plot_active_rate

# churn and Geography, Gender, NumOfProducts, HasCrCard and IsActiveMember (combined)
plot_combine1 <- grid.arrange(plot_country, plot_gender, plot_product,plot_crcard, plot_active, nrow = 3)
plot_combine1


#####Churn and credit core#####
plot_credit <- bank1 %>% 
  ggplot(aes(x = CreditScore, fill = Exited)) + 
  geom_density(alpha = 0.5) + 
  theme_minimal()
plot_credit

#####Churn and age#####
plot_age <- bank1 %>% 
  ggplot(aes(x = Age, fill = Exited)) + 
  geom_density(alpha = 0.5) 
  theme_minimal()
plot_age

#####Churn and tenure#####
plot_tenure <- bank1 %>% 
  ggplot(aes(x = Tenure, fill = Exited)) + 
  geom_density(alpha = 0.5) +
  theme_minimal()
plot_tenure

#####Churn and balance#####
plot_balance <- bank1 %>% 
  ggplot(aes(x = Balance, fill = Exited)) + 
  geom_density(alpha = 0.5) +
  theme_minimal()
plot_balance

#####Churn and estimated salary#####
plot_salary <- bank1 %>% 
  ggplot(aes(x = EstimatedSalary, fill = Exited)) + 
  geom_density(alpha = 0.5) +
  scale_x_sqrt() +
  theme_minimal()
plot_salary

# Churn and CreditScore, Age, Tenure, Balance and EstimatedSalary (combined)
plot_combine2 <- grid.arrange(plot_credit, plot_age, plot_tenure, plot_balance, plot_salary, nrow = 3 )
plot_combine2



####Model Building####

#####Create train set and test set from the big train set(bank1)#####
set.seed(5, sample.kind = "Rounding")
test_index_1 <- createDataPartition(y = bank1$Exited, times = 1, p = 0.1, list = FALSE)
test_set <- bank1[test_index_1,]
train_set <- bank1[-test_index_1,]

str(train_set)
str(test_set)



#####glm with caret#####

######Run a model######
modelLookup("glm")

control_glm <- trainControl(summaryFunction = prSummary,
                            classProbs = TRUE,
                            savePredictions = T,
                            method = "none"
                            )

set.seed(123, sample.kind = "Rounding")
train_glm <- train(Exited ~.,
                   data = train_set,
                   method = "glm",
                   trControl = control_glm)

pred_glm <- predict(train_glm, test_set, type = "prob")
pred_glm

######Find the best cutoff for highest F1 score######

glm_cutoffs <- tibble(obs = test_set$Exited,
                      pred = pred_glm$`yes`) %>%
  pr_curve(obs, pred) %>% 
  mutate(f1 = (2 * precision * recall) / (precision + recall)) %>% 
  arrange(-f1)

glm_best_cutoff <- pull(glm_cutoffs[1,1])
glm_best_cutoff #0.2836466

######Predict with the best cutoff and calculate the F1 score######

y_hat_glm <- factor(ifelse(pred_glm$`yes` >= glm_best_cutoff, "yes", "no"), levels = c("yes", "no"))

cm_glm <- confusionMatrix(data = y_hat_glm, 
                          reference = test_set$Exited,
                          mode = "prec_recall", positive = "yes")

glm_result <- tibble(Model = "glm",
                     Cutoff = glm_best_cutoff,
                     F1 = cm_glm$byClass["F1"],
                     Precision = cm_glm$byClass["Precision"],
                     Recall = cm_glm$byClass["Recall"])
glm_result



#####ranger with caret#####

modelLookup("ranger")

######Train a model with tuning parameters######

control_ranger <- trainControl(method = "cv",
                               number = 5,
                               p = 0.8,
                               summaryFunction = prSummary,
                               classProbs = TRUE,
                               savePredictions = T,
                               verboseIter = T,
                               allowParallel = T
                               )


set.seed(123, sample.kind = "Rounding")
train_ranger <- train(Exited ~.,
                      data = train_set,
                      method = "ranger",
                      num.trees = 500,
                      tuneGrid = expand.grid(mtry = 1:10,
                                             min.node.size = seq(10,50,10),
                                             splitrule = "gini"),
                      metric = "F",
                      trControl = control_ranger)

# Check the best tune parameter with automatic tuning
# The cutoff in this process is not the best

?predict.ranger

train_ranger$bestTune # mtry = 8, min.node.size = 20
train_ranger$results
train_ranger$pred

######Find the true best tune parameters manually######

# Firstly, calculate the F1 score of each tuning combination

pred_ranger <- train_ranger$pred

f1_scores_ranger <- pred_ranger %>% 
  group_by(mtry, min.node.size, Resample) %>% 
  pr_curve(obs, yes) %>% 
  mutate(f1 = (2 * precision * recall) / (precision + recall)) %>% 
  slice_max(f1, n = 1) %>% 
  group_by(mtry, min.node.size) %>% 
  summarise(f1 = mean(f1)) 

# Secondly, plot mtry against min.node.size, and look for the trend of F1 score

f1_scores_ranger %>% 
  ggplot(aes(mtry, min.node.size, color = f1, size = f1, label = f1)) +
  geom_point() +
  geom_text(aes(label=ifelse(f1>0.630,round(f1, digits = 3),'')),
            hjust=0,
            vjust=-1)

# Thirdly, get the parameters that yeild the highest F1 score

f1_scores_ranger_desc <- f1_scores_ranger %>% arrange(-f1)

best_mtry_ranger <- as.numeric(f1_scores_ranger_desc[1,1])
best_mtry_ranger # best mtry = 3

best_min.node.size_ranger <- as.numeric(f1_scores_ranger_desc[1,2])
best_min.node.size_ranger # best min.node.size = 10

######Re-run the model with the true best tune parameters######

Control_ranger_best <- trainControl(method = "none",
                                    summaryFunction = prSummary,
                                    classProbs = TRUE,
                                    savePredictions = T,
                                    verboseIter = T,
                                    allowParallel = T
                                    )

set.seed(123, sample.kind = "Rounding")
train_ranger_best <- train(Exited ~.,
                           data = train_set,
                           method = "ranger",
                           num.trees = 500,
                           tuneGrid = expand.grid(mtry = best_mtry_ranger,
                                                  min.node.size = best_min.node.size_ranger,
                                                  splitrule = "gini"),
                           metric = "F",
                           trControl = Control_ranger_best)

# Predict with the new model and get class probabilities

prob_ranger <- predict(train_ranger_best, test_set, type = 'prob')

######Find the best cutoff for highest F1 score######

ranger_cutoffs <- tibble(obs = test_set$Exited,
                         pred = prob_ranger$`yes`) %>%
  pr_curve(obs, pred) %>% 
  mutate(f1 = (2 * precision * recall) / (precision + recall)) %>% 
  arrange(-f1)

ranger_best_cutoff <- pull(ranger_cutoffs[1,1])
ranger_best_cutoff # 0.4035559

######Predict with the best cutoff and calculate the F1 score######

y_hat_ranger <- factor(ifelse(prob_ranger$`yes` >= ranger_best_cutoff, "yes", "no"), levels = c("yes", "no"))

cm_ranger <- confusionMatrix(data = y_hat_ranger,
                             reference = test_set$Exited,
                             mode = "prec_recall", positive = "yes")

ranger_result <- tibble(Model = "ranger",
                        Cutoff = ranger_best_cutoff,
                        F1 = cm_ranger$byClass["F1"],
                        Precision = cm_ranger$byClass["Precision"],
                        Recall = cm_ranger$byClass["Recall"])
ranger_result



#####knn with caret#####

######Train a model with tuning parameters######

modelLookup("knn")

control_knn <- trainControl(method = "cv",
                            number = 5,
                            p = 0.8,
                            summaryFunction = prSummary,
                            classProbs = TRUE,
                            savePredictions = T,
                            verboseIter = T,
                            allowParallel = T
                            )

set.seed(123, sample.kind = "Rounding")
train_knn <- train(Exited ~.,
                   data = train_set,
                   method = "knn",
                   preProcess = c("center", "scale"),
                   tuneGrid = data.frame(k = seq(3, 153, 2)),
                   metric = "F",
                   trControl = control_knn)

# Check the best tune parameter with automatic tuning
train_knn$bestTune # k = 3
train_knn$results
train_knn$pred

######Find the true best tune parameters manually######

# Firstly, calculate the F1 score of each k value

pred_knn <- train_knn$pred

f1_scores_knn <- pred_knn %>% 
  group_by(k, Resample) %>% 
  pr_curve(obs, yes) %>% 
  mutate(f1 = (2 * precision * recall) / (precision + recall)) %>% 
  slice_max(f1, n = 1) %>% 
  group_by(k) %>% 
  summarise(f1 = mean(f1)) 

# Secondly, plot k against f1, and look for the trend of F1 score

f1_scores_knn %>% 
  ggplot(aes(k, f1)) +
  geom_line()

# Thirdly, get the parameters that yeild the highest F1 score

f1_scores_knn_desc <- f1_scores_knn %>% arrange(-f1)

best_k_knn <- as.numeric(f1_scores_knn_desc[1,1])
best_k_knn # k = 63

######Re-run the model with the true best k######

control_knn_best <- trainControl(method = "none",
                                 summaryFunction = prSummary,
                                 classProbs = TRUE,
                                 savePredictions = T,
                                 verboseIter = T,
                                 allowParallel = T
                                 )

set.seed(123, sample.kind = "Rounding")
train_knn_best <- train(Exited ~.,
                        data = train_set,
                        method = "knn",
                        preProcess = c("center", "scale"),
                        tuneGrid = data.frame(k = best_k_knn),
                        metric = "F",
                        trControl = control_knn_best)

prob_knn <- predict(train_knn_best, test_set, type = 'prob')

######Find the best cutoff for highest F1 score######

knn_cutoffs <- tibble(obs = test_set$Exited,
                      pred = prob_knn$`yes`) %>%
  pr_curve(obs, pred) %>% 
  mutate(f1 = (2 * precision * recall) / (precision + recall)) %>% 
  arrange(-f1)

knn_best_cutoff <- pull(knn_cutoffs[1,1])
knn_best_cutoff # 0.2698413

######Predict with the best cutoff and calculate the F1 score######

y_hat_knn <- factor(ifelse(prob_knn$`yes` >= knn_best_cutoff, "yes", "no"), levels = c("yes", "no"))

cm_knn <- confusionMatrix(data = y_hat_knn,
                          reference = test_set$Exited,
                          mode = "prec_recall", positive = "yes")

knn_result <- tibble(Model = "knn",
                     Cutoff = knn_best_cutoff,
                     F1 = cm_knn$byClass["F1"],
                     Precision = cm_knn$byClass["Precision"],
                     Recall = cm_knn$byClass["Recall"])
knn_result



#####LightGBM#####

######Prepare data sets for LightGBM format######

# Transform factor variables to numeric

train_set_lgbm <- train_set
train_set_lgbm$Geography <- as.numeric(train_set_lgbm$Geography)
train_set_lgbm$Gender <- as.numeric(train_set_lgbm$Gender)
train_set_lgbm$HasCrCard <- as.numeric(train_set_lgbm$HasCrCard) - 1
train_set_lgbm$IsActiveMember <- as.numeric(train_set_lgbm$IsActiveMember) - 1
train_set_lgbm$Exited <- ifelse(as.numeric(train_set_lgbm$Exited) == 2, 0, 1)


test_set_lgbm <- test_set
test_set_lgbm$Geography <- as.numeric(test_set_lgbm$Geography)
test_set_lgbm$Gender <- as.numeric(test_set_lgbm$Gender)
test_set_lgbm$HasCrCard <- as.numeric(test_set_lgbm$HasCrCard) - 1
test_set_lgbm$IsActiveMember <- as.numeric(test_set_lgbm$IsActiveMember) - 1
test_set_lgbm$Exited <- ifelse(as.numeric(test_set_lgbm$Exited) == 2, 0, 1)

# Define catogorical features

categoricals_vector  <- c(2, 3, 8, 9)

# Split train set and test set into data sets that contain feature and label.
# Transform the data sets to matrix

train_set_x <- as.matrix(train_set_lgbm[, -11])
train_set_y <- as.matrix(train_set_lgbm[, 11])

test_set_x <- as.matrix(test_set_lgbm[, -11])
test_set_y <- as.matrix(test_set_lgbm[, 11])

# Load the train and test data into the LightGBM dataset object

dtrain <- lgb.Dataset(data = train_set_x, 
                      label = train_set_y, 
                      params = list(categorical_feature = categoricals_vector))

dtest <- lgb.Dataset.create.valid(dtrain, 
                                  data = test_set_x, 
                                  label = test_set_y, 
                                  params = list(categorical_feature = categoricals_vector))

######Train a model######

# Define parameters

train_params <- list(
  objective= 'binary',
  is_unbalance = TRUE,
  metric = "average_precision",
  num_iterations = 100,
  learning_rate = 0.1
  ) 

# Cross Validation

set.seed(123, sample.kind = "Rounding")
lgm_cv <- lgb.cv(params = train_params,
                 data = dtrain,
                 nfold = 5,
                 stratified = TRUE)

best_iter <- lgm_cv$best_iter
best_iter #62

# Run the model with best_iter
set.seed(123, sample.kind = "Rounding")
train_lgbm <- lgb.train(params = train_params,
                        data = dtrain,
                        nrounds = best_iter,
                        verbose = 2)

######Find the best cutoff for highest F1 score######

pred_lgbm <- predict(train_lgbm, test_set_x)

lgbm_cutoffs <- tibble(obs = test_set$Exited,
                       pred = pred_lgbm) %>% 
  pr_curve(obs, pred) %>% 
  mutate(f1 = (2 * precision * recall) / (precision + recall)) %>% 
  arrange(-f1)

lgbm_best_cutoff <- pull(lgbm_cutoffs[1,1])
lgbm_best_cutoff #0.6324961

######Predict with the best cutoff and calculate F1 score######
y_hat_lgbm <- factor(ifelse(pred_lgbm >= lgbm_best_cutoff, 1, 0), levels = c(0, 1))

cm_lgbm <- confusionMatrix(data = y_hat_lgbm,
                           reference = factor(test_set_lgbm$Exited, levels = c(0, 1)),
                           mode = "prec_recall", positive = "1")

lgbm_result <- tibble(Model = "LightGBM",
                      Cutoff = lgbm_best_cutoff,
                      F1 = cm_lgbm$byClass["F1"],
                      Precision = cm_lgbm$byClass["Precision"],
                      Recall = cm_lgbm$byClass["Recall"])
lgbm_result



####Final Validation with ranger####

# Use "bank1" as train set, and "validation" as test set.

control_final <- trainControl(method = "none",
                              summaryFunction = prSummary,
                              classProbs = TRUE,
                              savePredictions = T,
                              verboseIter = T,
                              allowParallel = T
                              )

set.seed(123, sample.kind = "Rounding")
train_ranger_final <- train(Exited ~.,
                            data = bank1,
                            method = "ranger",
                            num.trees = 500,
                            tuneGrid = expand.grid(mtry = best_mtry_ranger,
                                                   min.node.size = best_min.node.size_ranger,
                                                   splitrule = "gini"),
                            metric = "F",
                            trControl = control_final)

prob_ranger_final <- predict(train_ranger_final, validation, type = 'prob')

# Predict with the best cutoff and calculate the F1 score

y_hat_final <- factor(ifelse(prob_ranger_final$`yes` >= ranger_best_cutoff, "yes", "no"), levels = c("yes", "no"))

cm_final <- confusionMatrix(data = y_hat_final,
                            reference = validation$Exited,
                            mode = "prec_recall", positive = "yes")

final_validation_result <- tibble(Model = "ranger",
                                  Cutoff = ranger_best_cutoff,
                                  F1 = cm_final$byClass["F1"],
                                  Precision = cm_final$byClass["Precision"],
                                  Recall = cm_final$byClass["Recall"])
final_validation_result



