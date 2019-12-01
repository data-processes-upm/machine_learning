# Motivation: predict cancer malignancy based on image data

# Set up
library(dplyr)
library(ggplot2)
library(tibble)
library(caret)
library(e1071) # dependency for KNN

# Load data, downloaded from:
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
cancer <- read.csv("data/breast_cancer.csv", stringsAsFactors = F)

# Make `diagnosis` a factor variable (required for some ml functions)
cancer <- cancer %>%
  mutate(diagnosis = factor(diagnosis))

# Set "seed" so all random values are the same across computers
set.seed(42)

# What is the distribution of `diagnosis` (malignant v.s. benign cases)?
table(cancer$diagnosis)


# Which variable is most correlated with the outcome?
# Hints:
# - convert `diagnosis` to a numeric variable
# - use the `cor()` function
# - select/view results of interest from the correlation matrix
corrs <- cancer %>%
  mutate(diagnosis_binary = if_else(diagnosis == "M", 0, 1)) %>%
  select(-diagnosis) %>%
  cor() %>%
  data.frame() %>%
  select(corr = diagnosis_binary) %>%
  rownames_to_column("variable") %>%
  filter(variable != "diagnosis_binary") %>%
  arrange(corr) %>%
  mutate(variable = factor(variable, levels = variable))

# Draw overlapping histograms of the most correlated feature (by outcome)
ggplot(data = cancer) +
  geom_histogram(
    mapping = aes(x = concave.points_worst, fill = diagnosis),
    alpha = .7,
    position = "identity"
  )

# Time pending: plot the correlation values in a way you find
# meaningful and expressive
ggplot(data = corrs) +
  geom_point(mapping = aes(x = variable, y = corr)) +
  coord_flip() +
  geom_hline(yintercept = 0) +
  ylim(-.8, .8) +
  ggtitle("Correlation between features and malignancy") +
  labs(x = "Feature", y = "Correlation")

# -------- Split into training/testing sets --------
# See: http://topepo.github.io/caret/data-splitting.html

# Create a set of training indices
trainIndex <- createDataPartition(cancer$diagnosis,
  p = .8,
  list = FALSE,
  times = 1
)

# Subset your data into training and testing set
training_set <- cancer[ trainIndex, ]
test_set <- cancer[ -trainIndex, ]

# -------- First approach: fit a KNN model --------
# Create a `knn` fit, training on the *training* data
basic_fit <- train(diagnosis ~ ., data = training_set, method = "knn")

# Use your model to make predictions on your *test* data
basic_preds <- predict(basic_fit, test_set)

# Assess your performance by creating a confusion matrix of your predcitions
confusionMatrix(test_set$diagnosis, basic_preds, positive = "M")

# -------- Second approach: add 10 fold cross validation to your training ------
# Specify cross validation approach: 10 fold CV
# See: http://topepo.github.io/caret/model-training-and-tuning.html#basic-parameter-tuning
fitControl <- trainControl(
  method = "cv",
  number = 10
)

# Train, specifying cross validation
fit_with_cv <- train(
  diagnosis ~ .,
  data = training_set,
  method = "knn",
  trControl = fitControl
)


# Make predictions on the test set
fit_cv_preds <- predict(fit_with_cv, test_set)

# Assess performance via a confusion matrix
confusionMatrix(test_set$diagnosis, fit_cv_preds, positive = "M")

# -------- Third approach: add a grid search to find the best value for k ------

# Create a grid of parameters to search through (values for k, 1:20)
grid <- expand.grid(k = 1:20)

# Fit with grid search
fit_cv_grid <- train(
  diagnosis ~ .,
  data = training_set,
  method = "knn",
  trControl = fitControl,
  tuneGrid = grid
)

# Make predictions on the test set
preds_cv_grid <- predict(fit_cv_grid, test_set)

# Assess performance via a confusion matrix
confusionMatrix(test_set$diagnosis, preds_cv_grid, positive = "M")

# Show the average performance (across folds) for each value of `k`
ggplot(data = preds_cv_grid$results) +
  geom_line(mapping = aes(x = k, y = Accuracy))

# -------- Fourth approach: add a preprocessing step to normalize features -----
# Train, adding a `preProcess` step, setting the argument to "range"
fit_cv_grid_pp <- train(
  diagnosis ~ .,
  data = training_set,
  method = "knn",
  trControl = fitControl,
  tuneGrid = grid,
  preProcess = "range"
)

# Make predictions on the test set
fit_cv_grid_pp_preds <- predict(fit_cv_grid_pp, test_set)

# Assess performance via a confusion matrix
confusionMatrix(test_set$diagnosis, fit_cv_grid_pp_preds, positive = "M")

# Show the average performance (across folds) for each value of `k`
ggplot(data = fit_cv_grid_pp$results) +
  geom_line(mapping = aes(x = k, y = Accuracy))

# An aside: nice explanation of "kappa" values
# https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
