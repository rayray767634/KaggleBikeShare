library(vroom)
library(tidyverse)
library(DataExplorer)
library(dplyr)
library(GGally)
library(patchwork)

# reading in data
bike.test <- vroom("test.csv")
bike.train <- vroom("train.csv")

#change columns to factors
bike.train$season <- as.factor(bike.train$season)
bike.train$holiday <- as.factor(bike.train$holiday)
bike.train$workingday <- as.factor(bike.train$workingday)
bike.train$weather <- as.factor(bike.train$weather)

# ggplots
bike.reg.plot <- ggplot(data = bike.train, mapping = aes(x = registered, y = count)) +
  geom_point()

bike.temp.plot <- ggplot(data = bike.train, mapping = aes(x = temp, y = count)) +
  geom_point()

bike.weather.bar.plot <- ggplot(data = bike.train, mapping = aes(x = weather)) +
  geom_bar()

#DataExplorer plots
plot_intro(bike.train)
bike.cor.plot <- plot_correlation(bike.train)
bike.bar.plot <- plot_bar(bike.train)
bike.hist.plot <- plot_histogram(bike.train)

# making a 4 panel plot
(bike.cor.plot + bike.hist.plot) / (bike.temp.plot + bike.bar.plot)
