library(vroom)
library(tidyverse)
library(DataExplorer)
library(dplyr)
library(GGally)
library(patchwork)
library(tidymodels)
library(lubridate)

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

## Get Predictions for test set AND format for Kaggle
test_preds <- predict(bike_workflow, new_data = bike.test) %>%
  bind_cols(., bike.test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
# make a csv of predictions

vroom_write(x=test_preds, file="./BikePreds.csv", delim=",")
