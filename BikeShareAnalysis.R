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
bike.train <- vroom("train.csv")

# move weather = 4 to weather = 3
bike.train <- bike.train %>%
  mutate(weather = ifelse(weather == 4,3,weather))

# feature engineering

my_recipe <- recipe(count~., data = bike.train) %>%
  # make these variables factors
  step_mutate(season=factor(season)) %>%
  step_mutate(holiday=factor(holiday)) %>%
  step_mutate(workingday=factor(workingday)) %>%
  step_mutate(weather=factor(weather)) %>%
  step_time(datetime,features = "hour") %>% # add hour variable
  # remove variables
  step_rm(casual) %>%
  step_rm(registered) %>%
  # change name of variable
  step_rename(hour = datetime_hour)

# prepare and bake recipe
prepped_recipe <- prep(my_recipe)
bike.train.clean <- bake(prepped_recipe, new_data = NULL)

bike.train.clean
