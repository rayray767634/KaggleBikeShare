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
library(stacks)
library(dbarts)
library(sparklyr)

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

# Tuning Models

preg_model2 <- linear_reg(penalty = tune(),
                          mixture = tune()) %>% # Set model and tuning
  set_engine("glmnet") # Function to fit in R

# Set workflow
preg_wf2 <- workflow() %>%
  add_recipe(my_recipe2) %>%
  add_model(preg_model2)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 10) 

# Split data for CV
folds <- vfold_cv(logTrainSet, v = 20, repeats = 1)

# Run the CV
CV_results <- preg_wf2 %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq))

# plot results
collect_metrics(CV_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(data=., aes(x = penalty, y = mean, color = factor(mixture))) +
  geom_line()

# find best tuning parameters

bestTune <- CV_results %>%
  select_best("rmse")

# Finalize the workflow and fit it
final_wf <- 
  preg_wf2 %>%
  finalize_workflow(bestTune) %>%
  fit(data = logTrainSet)

# Predict
bike_preds_tuned <- final_wf %>%
  predict(new_data = bike.test) %>%
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike.test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=bike_preds_tuned, file="./PregBikePreds2.csv", delim=",")


## regression tree

my_mod3 <- decision_tree(tree_depth = tune(),
                         cost_complexity = tune(),
                         min_n = tune()) %>% # Type of model
  set_engine("rpart") %>% # Engine = What r function to use
  set_mode("regression")

# Create a workflow with model and recipe
my_recipe3 <- recipe(count~., data = logTrainSet) %>%
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


wf2 <- workflow() %>%
  add_recipe(my_recipe3) %>%
  add_model(my_mod3)

# Set up grid of tuning values
tuning_grid2 <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 5)

# Set up K-fold CV
folds <- vfold_cv(logTrainSet, v = 5, repeats = 1)
# Run the CV
CV_results2 <- wf2 %>%
  tune_grid(resamples = folds,
            grid = tuning_grid2,
            metrics = metric_set(rmse, mae, rsq))

# Find best tuning parameters
bestTune2 <- CV_results2 %>%
  select_best("rmse")

# Finalize workflow and predict
final_wf2 <- 
  wf2 %>%
  finalize_workflow(bestTune2) %>%
  fit(data = logTrainSet)

bike_preds_tree <- final_wf2 %>%
  predict(new_data = bike.test) %>%
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike.test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=bike_preds_tree, file="./BikePredsTree.csv", delim=",")


## random forest

my_mod4 <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>% #Type of model
  set_engine("ranger") %>% # what r function to use
  set_mode("regression")

# Create a workflow with model and recipe
my_recipe3 <- recipe(count~., data = logTrainSet) %>%
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


wf3 <- workflow() %>%
  add_recipe(my_recipe3) %>%
  add_model(my_mod4)

# Set up grid of tuning values
tuning_grid3 <- grid_regular(mtry(range = c(1,10)),
                             min_n(),
                             levels = 5)

# Set up K-fold CV
folds <- vfold_cv(logTrainSet, v = 5, repeats = 1)
# Run the CV
CV_results3 <- wf3 %>%
  tune_grid(resamples = folds,
            grid = tuning_grid3,
            metrics = metric_set(rmse, mae, rsq))

# Find best tuning parameters
bestTune3 <- CV_results3 %>%
  select_best("rmse")

# Finalize workflow and predict
final_wf3 <- 
  wf3 %>%
  finalize_workflow(bestTune3) %>%
  fit(data = logTrainSet)

bike_preds_forest <- final_wf3 %>%
  predict(new_data = bike.test) %>%
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike.test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=bike_preds_forest, file="./BikePredsForest.csv", delim=",")


# model stacking
untunedModel <- control_stack_grid() # set of models
tunedModel <- control_stack_resamples() # single model that is already done

# define the model
lin_model <- linear_reg() %>% 
  set_engine("lm")

# set up the whole workflow
linreg_wf <- workflow() %>%
  add_recipe(my_recipe3) %>%
  add_model(lin_model)

# fit linear regression to folds
linreg_folds_fit <- linreg_wf %>%
  fit_resamples(resamples = folds,
                metric_set(rmse),
                control = tunedModel)

# penalized regression
# define the model
pen_reg_model <- linear_reg(mixture = tune(),
                            penalty = tune()) %>%
  set_engine("glmnet")

# define the workflow

pen_reg_wf <- workflow() %>%
  add_recipe(my_recipe3) %>%
  add_model(pen_reg_model)

# define a regular grid

penReg_tuneGrid <- grid_regular(mixture(),
                                penalty(),
                                levels = 5)

# fit to Folds
penReg_folds_fit <- pen_reg_wf %>%
  tune_grid(resamples = folds,
            grid = penReg_tuneGrid,
            metrics = metric_set(rmse),
            control = untunedModel)

# tree
reg_tree <- decision_tree(tree_depth = tune(),
                          cost_complexity = tune(),
                          min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

# workflow
regTree_wf <- workflow() %>%
  add_recipe(my_recipe3) %>%
  add_model(reg_tree)

# tuning Grid
regTree_tuneGrid <- grid_regular(tree_depth(),
                                 cost_complexity(),
                                 min_n(),
                                 levels = 5)

# tune the model
trees_folds_fit <- regTree_wf %>%
  tune_grid(resamples = folds,
            grid = regTree_tuneGrid,
            metrics = metric_set(rmse),
            control = untunedModel)

# stack the models together

bike_stack <- stacks() %>%
  add_candidates(linreg_folds_fit) %>%
  add_candidates(penReg_folds_fit) %>%
  add_candidates(trees_folds_fit)

as_tibble(bike_stack)

fitted_bike_stack <- bike_stack %>%
  blend_predictions() %>%
  fit_members()

bike_preds_stack <- predict(fitted_bike_stack,new_data = bike.test) %>%
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike.test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=bike_preds_stack, file="./BikePredsStack.csv", delim=",")

# bart

my_mod6 <- 
  parsnip::bart(
    trees = 100,
    prior_terminal_node_coef = .95,
    prior_terminal_node_expo = 2
  ) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression")

# Create a workflow with model and recipe
my_recipe3 <- recipe(count~., data = logTrainSet) %>%
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


wf5 <- workflow() %>%
  add_recipe(my_recipe3) %>%
  add_model(my_mod6)

# Set up grid of tuning values
tuning_grid5 <- grid_regular(trees(),
                             prior_terminal_node_coef(),
                             prior_terminal_node_expo())

# Set up K-fold CV
folds <- vfold_cv(logTrainSet, v = 2, repeats = 1)
# Run the CV
CV_results5 <- wf5 %>%
  tune_grid(resamples = folds,
            grid = tuning_grid5,
            metrics = metric_set(rmse, mae, rsq))

# Find best tuning parameters
bestTune5 <- CV_results5 %>%
  select_best("rmse")

# Finalize workflow and predict
final_wf5 <- 
  wf5 %>%
  finalize_workflow(bestTune5) %>%
  fit(data = logTrainSet)

bike_preds_bart <- final_wf5 %>%
  predict(new_data = bike.test) %>%
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike.test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=bike_preds_bart, file="./BikePredsBart.csv", delim=",")

