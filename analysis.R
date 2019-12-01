# Motivation: predict cancer malignancy based on image data

# Set up
library(dplyr)
library(ggplot2)
library(tibble)
library(caret)
library(e1071) # dependency for KNN

# Load data, downloaded from:
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

# Make `diagnosis` a factor variable (required for some ml functions)

# Set "seed" so all random values are the same across computers

# What is the distribution of `diagnosis` (malignant v.s. benign cases)?

# Which variable is most correlated with the outcome?
# Hints:
# - convert `diagnosis` to a numeric variable
# - use the `cor()` function
# - select/view results of interest from the correlation matrix

# Draw overlapping histograms of the most correlated feature (by outcome)

# Time pending: plot the correlation values in a way you find
# meaningful and expressive

# -------- Split into training/testing sets --------
# See: http://topepo.github.io/caret/data-splitting.html

# Create a set of training indices

# Subset your data into training and testing set

# -------- First approach: fit a KNN model --------
# Create a `knn` fit, training on the *training* data

# Use your model to make predictions on your *test* data

# Assess your performance by creating a confusion matrix of your predcitions

# -------- Second approach: add 10 fold cross validation to your training ------
# Specify cross validation approach: 10 fold CV
# See: http://topepo.github.io/caret/model-training-and-tuning.html#basic-parameter-tuning

# Train, specifying cross validation

# Make predictions on the test set

# Assess performance via a confusion matrix

# -------- Third approach: add a grid search to find the best value for k ------

# Create a grid of parameters to search through (values for k, 1:20)

# Fit with grid search

# Make predictions on the test set

# Assess performance via a confusion matrix

# Show the average performance (across folds) for each value of `k`

# -------- Fourth approach: add a preprocessing step to normalize features -----
# Train, adding a `preProcess` step, setting the argument to "range"

# Make predictions on the test set

# Assess performance via a confusion matrix

# Show the average performance (across folds) for each value of `k`

# An aside: nice explanation of "kappa" values
# https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
