library(vroom)
library(tidyverse)
library(DataExplorer)
library(dplyr)
library(GGally)
library(patchwork)
library(tidymodels)
library(lubridate)
library(poissonreg)
library(glmnet)

# reading in data
bike.test <- vroom("test.csv") 

bike.train <- vroom("train.csv") %>% 
  select(-c("casual","registered"))

# feature engineering

my_recipe <- recipe(count~., data = bike.train) %>%
  # make these variables factors
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime,features = "hour") %>% # add hour variable
  # change name of variable
  step_rename(hour = datetime_hour) %>%
  step_rm(datetime)

# prepare and bake recipe
prepped_recipe <- prep(my_recipe) 
bake(prepped_recipe, new_data = bike.train)
bake(prepped_recipe, new_data = bike.test)


# linear regression

my_mod <- linear_reg() %>%
  set_engine("lm")

bike_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bike.train)

extract_fit_engine(bike_workflow) %>%
  tidy()

## Get Predictions for test set AND format for Kaggle
test_preds <- predict(bike_workflow, new_data = bike.test) %>%
  bind_cols(., bike.test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
# make a csv of predictions

vroom_write(x=test_preds, file="./BikePreds.csv", delim=",")


# Poisson Regression

pois_mod <- poisson_reg() %>% # type of model
  set_engine("glm") # GLM = generalized linear model

bike_pois_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>%
  add_model(pois_mod) %>%
  fit(data = bike.train) # fit the workflow

bike_predictions <- predict(bike_pois_workflow, new_data = bike.test) %>%
  bind_cols(., bike.test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=bike_predictions, file="./PoissonBikePreds.csv", delim=",")

# penalized regression

logTrainSet <- bike.train %>%
  mutate(count=log(count))

my_recipe2 <- recipe(count~., data = logTrainSet) %>%
  # make these variables factors
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime,features = "hour") %>% # add hour variable
  # change name of variable
  step_rename(hour = datetime_hour) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

preg_model <- linear_reg(penalty = .0000000001, mixture = 0) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R
preg_wf <- workflow() %>%
  add_recipe(my_recipe2) %>%
  add_model(preg_model) %>%
  fit(data = logTrainSet)

bike_predictions_preg <- predict(preg_wf, new_data = bike.test) %>%
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike.test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=bike_predictions_preg, file="./PregBikePreds.csv", delim=",")
