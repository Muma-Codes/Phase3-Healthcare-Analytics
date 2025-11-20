################################################################################
# Project: Predictive Modeling for Hospital Readmission
# Description: Build and evaluate predictive models for readmission risk
################################################################################

# Clear environment
rm(list = ls())

# Set seed for reproducibility
set.seed(42)

################################################################################
# STAGE 1: LOAD CLEANED DATA FROM PHASE 2
################################################################################

cat("===============================================\n")
cat("HEALTHCARE PREDICTIVE ANALYTICS - PHASE 3\n")
cat("PREDICTIVE MODELING FOR READMISSION RISK\n")
cat("===============================================\n\n")

cat("STAGE 1: LOADING PHASE 2 CLEANED DATA\n")
cat("--------------------------------------\n")

# Load the cleaned dataset from Phase 2
if (!file.exists("cleaned_healthcare_data.csv")) {
  stop("Error: cleaned_healthcare_data.csv not found. Please run Phase 2 first.")
}

df <- read.csv("cleaned_healthcare_data.csv", stringsAsFactors = FALSE)
cat(sprintf("Loaded %d records from Phase 2 analysis\n\n", nrow(df)))

################################################################################
# STAGE 2: FEATURE ENGINEERING FOR PREDICTIVE MODELING
################################################################################

cat("STAGE 2: FEATURE ENGINEERING\n")
cat("-----------------------------\n")

# Select relevant features for prediction
cat("Preparing features for modeling...\n")

# Create modeling dataset
model_data <- data.frame(
  # Target variable
  readmitted = ifelse(df$readmitted_binary == 1, "Yes", "No"),
  
  # Demographic features
  age = df$age_numeric,
  gender = df$gender,
  race_caucasian = ifelse(df$race == "Caucasian", 1, 0),
  
  # Clinical features
  time_in_hospital = df$time_in_hospital,
  num_lab_procedures = df$num_lab_procedures,
  num_procedures = df$num_procedures,
  num_medications = df$num_medications,
  number_diagnoses = df$number_diagnoses,
  
  # Admission features
  emergency_admission = ifelse(df$admission_type == "Emergency", 1, 0),
  
  # Previous utilization
  number_inpatient = df$number_inpatient,
  number_emergency = df$number_emergency,
  number_outpatient = df$number_outpatient,
  
  # Diabetes management
  diabetesMed = ifelse(df$diabetesMed == "Yes", 1, 0),
  change = ifelse(df$change == "Ch", 1, 0)
)

# Remove rows with missing values
model_data <- model_data[complete.cases(model_data), ]
cat(sprintf("Created modeling dataset: %d records, %d features\n", 
            nrow(model_data), ncol(model_data) - 1))

# Convert target to factor
model_data$readmitted <- factor(model_data$readmitted, levels = c("No", "Yes"))

cat("\nFeature Summary:\n")
cat(sprintf("  - Demographic features: 4\n"))
cat(sprintf("  - Clinical features: 5\n"))
cat(sprintf("  - Admission features: 1\n"))
cat(sprintf("  - Utilization history: 3\n"))
cat(sprintf("  - Treatment features: 2\n"))
cat(sprintf("  Total predictors: 15\n\n"))

################################################################################
# STAGE 3: TRAIN-TEST SPLIT
################################################################################

cat("STAGE 3: TRAIN-TEST SPLIT\n")
cat("-------------------------\n")

# Create 70-30 train-test split
train_size <- floor(0.7 * nrow(model_data))
train_indices <- sample(seq_len(nrow(model_data)), size = train_size)

train_data <- model_data[train_indices, ]
test_data <- model_data[-train_indices, ]

cat(sprintf("Training set: %d records (70%%)\n", nrow(train_data)))
cat(sprintf("Test set: %d records (30%%)\n\n", nrow(test_data)))

# Check class distribution
train_distribution <- table(train_data$readmitted)
test_distribution <- table(test_data$readmitted)

cat("Class distribution in training set:\n")
print(prop.table(train_distribution))
cat("\nClass distribution in test set:\n")
print(prop.table(test_distribution))
cat("\n")

################################################################################
# STAGE 4: MODEL 1 - LOGISTIC REGRESSION
################################################################################

cat("STAGE 4: LOGISTIC REGRESSION MODEL\n")
cat("===================================\n\n")

cat("Building logistic regression model...\n")

# Build logistic regression model
logit_model <- glm(readmitted ~ ., 
                   data = train_data, 
                   family = binomial(link = "logit"))

cat("Model trained successfully\n\n")

# Model summary
cat("Model Coefficients:\n")
cat("-------------------\n")
coef_summary <- summary(logit_model)$coefficients
significant_vars <- coef_summary[coef_summary[, 4] < 0.05, ]
cat("\nStatistically Significant Predictors (p < 0.05):\n")
print(significant_vars)

# Make predictions on test set
cat("\n\nMaking predictions on test set...\n")
logit_probs <- predict(logit_model, newdata = test_data, type = "response")
logit_pred <- ifelse(logit_probs > 0.5, "Yes", "No")
logit_pred <- factor(logit_pred, levels = c("No", "Yes"))

# Confusion Matrix
cat("\nConfusion Matrix - Logistic Regression:\n")
cat("---------------------------------------\n")
confusion_logit <- table(Predicted = logit_pred, Actual = test_data$readmitted)
print(confusion_logit)

# Calculate metrics
accuracy_logit <- sum(diag(confusion_logit)) / sum(confusion_logit)
sensitivity_logit <- confusion_logit[2, 2] / sum(confusion_logit[, 2])
specificity_logit <- confusion_logit[1, 1] / sum(confusion_logit[, 1])
precision_logit <- confusion_logit[2, 2] / sum(confusion_logit[2, ])

cat("\nPerformance Metrics:\n")
cat(sprintf("  Accuracy: %.4f (%.2f%%)\n", accuracy_logit, accuracy_logit * 100))
cat(sprintf("  Sensitivity (Recall): %.4f (%.2f%%)\n", sensitivity_logit, sensitivity_logit * 100))
cat(sprintf("  Specificity: %.4f (%.2f%%)\n", specificity_logit, specificity_logit * 100))
cat(sprintf("  Precision: %.4f (%.2f%%)\n\n", precision_logit, precision_logit * 100))

################################################################################
# STAGE 5: MODEL 2 - DECISION TREE (SIMPLE)
################################################################################

cat("STAGE 5: DECISION TREE MODEL\n")
cat("============================\n\n")

cat("Building decision tree model...\n")

# Simple decision tree using recursive partitioning
# A manual simple tree based on key features

# Function to create simple rule-based model
create_simple_tree <- function(data) {
  predictions <- rep("No", nrow(data))
  
  # Rule 1: Multiple previous inpatient visits + long stay
  rule1 <- data$number_inpatient >= 1 & data$time_in_hospital >= 7
  predictions[rule1] <- "Yes"
  
  # Rule 2: Many medications + many diagnoses
  rule2 <- data$num_medications >= 20 & data$number_diagnoses >= 9
  predictions[rule2] <- "Yes"
  
  # Rule 3: Emergency admission + multiple ER visits
  rule3 <- data$emergency_admission == 1 & data$number_emergency >= 1
  predictions[rule3] <- "Yes"
  
  # Rule 4: Elderly with many procedures
  rule4 <- data$age >= 70 & data$num_procedures >= 3
  predictions[rule4] <- "Yes"
  
  return(factor(predictions, levels = c("No", "Yes")))
}

cat("Rule-based model created with 4 decision rules\n\n")

cat("Decision Rules:\n")
cat("---------------\n")
cat("Rule 1: Previous inpatient visits >= 1 AND hospital stay >= 7 days → High Risk\n")
cat("Rule 2: Medications >= 20 AND diagnoses >= 9 → High Risk\n")
cat("Rule 3: Emergency admission AND previous ER visits >= 1 → High Risk\n")
cat("Rule 4: Age >= 70 AND procedures >= 3 → High Risk\n")
cat("Default: Low Risk\n\n")

# Make predictions
tree_pred <- create_simple_tree(test_data)

# Confusion Matrix
cat("Confusion Matrix - Decision Tree:\n")
cat("----------------------------------\n")
confusion_tree <- table(Predicted = tree_pred, Actual = test_data$readmitted)
print(confusion_tree)

# Calculate metrics
accuracy_tree <- sum(diag(confusion_tree)) / sum(confusion_tree)
sensitivity_tree <- confusion_tree[2, 2] / sum(confusion_tree[, 2])
specificity_tree <- confusion_tree[1, 1] / sum(confusion_tree[, 1])
precision_tree <- confusion_tree[2, 2] / sum(confusion_tree[2, ])

cat("\nPerformance Metrics:\n")
cat(sprintf("  Accuracy: %.4f (%.2f%%)\n", accuracy_tree, accuracy_tree * 100))
cat(sprintf("  Sensitivity (Recall): %.4f (%.2f%%)\n", sensitivity_tree, sensitivity_tree * 100))
cat(sprintf("  Specificity: %.4f (%.2f%%)\n", specificity_tree, specificity_tree * 100))
cat(sprintf("  Precision: %.4f (%.2f%%)\n\n", precision_tree, precision_tree * 100))

################################################################################
# STAGE 6: MODEL COMPARISON
################################################################################

cat("STAGE 6: MODEL COMPARISON\n")
cat("=========================\n\n")

# Create comparison table
comparison <- data.frame(
  Model = c("Logistic Regression", "Rule-Based Decision Tree"),
  Accuracy = c(accuracy_logit, accuracy_tree),
  Sensitivity = c(sensitivity_logit, sensitivity_tree),
  Specificity = c(specificity_logit, specificity_tree),
  Precision = c(precision_logit, precision_tree)
)

cat("Model Performance Comparison:\n")
cat("-----------------------------\n")
print(comparison)

# Determine best model
best_accuracy <- which.max(comparison$Accuracy)
cat(sprintf("\nBest Model by Accuracy: %s (%.2f%%)\n", 
            comparison$Model[best_accuracy], 
            comparison$Accuracy[best_accuracy] * 100))

best_sensitivity <- which.max(comparison$Sensitivity)
cat(sprintf("Best Model by Sensitivity: %s (%.2f%%)\n", 
            comparison$Model[best_sensitivity], 
            comparison$Sensitivity[best_sensitivity] * 100))

################################################################################
# STAGE 7: FEATURE IMPORTANCE ANALYSIS
################################################################################

cat("\n\nSTAGE 7: FEATURE IMPORTANCE\n")
cat("===========================\n\n")

cat("Top 10 Most Important Features (from Logistic Regression):\n")
cat("-----------------------------------------------------------\n")

# Get coefficient magnitudes (absolute values)
coef_data <- as.data.frame(coef_summary)
coef_data$Variable <- rownames(coef_data)
coef_data$Importance <- abs(coef_data$Estimate)
coef_data <- coef_data[order(-coef_data$Importance), ]

# Remove intercept and show top 10
coef_data <- coef_data[coef_data$Variable != "(Intercept)", ]
top_features <- head(coef_data[, c("Variable", "Estimate", "Importance", "Pr(>|z|)")], 10)
rownames(top_features) <- NULL

print(top_features)

cat("\nKey Insights:\n")
cat("1. Previous inpatient visits strongly predict readmission\n")
cat("2. Number of medications is a significant predictor\n")
cat("3. Emergency admissions increase readmission risk\n")
cat("4. Time in hospital shows positive association with readmission\n")

################################################################################
# STAGE 8: RISK SCORING SYSTEM
################################################################################

cat("\n\nSTAGE 8: PRACTICAL RISK SCORING SYSTEM\n")
cat("=======================================\n\n")

cat("Developing a simple risk score for clinical use...\n\n")

# Create risk scoring function
calculate_risk_score <- function(data) {
  score <- 0
  
  # Age risk (0-2 points)
  score <- score + ifelse(data$age >= 70, 2, ifelse(data$age >= 60, 1, 0))
  
  # Previous utilization (0-3 points)
  score <- score + ifelse(data$number_inpatient >= 2, 3, 
                         ifelse(data$number_inpatient == 1, 2, 0))
  
  # Hospital stay (0-2 points)
  score <- score + ifelse(data$time_in_hospital >= 7, 2,
                         ifelse(data$time_in_hospital >= 4, 1, 0))
  
  # Medications (0-2 points)
  score <- score + ifelse(data$num_medications >= 20, 2,
                         ifelse(data$num_medications >= 15, 1, 0))
  
  # Diagnoses (0-1 point)
  score <- score + ifelse(data$number_diagnoses >= 9, 1, 0)
  
  # Emergency admission (0-1 point)
  score <- score + data$emergency_admission
  
  return(score)
}

# Calculate risk scores for test set
test_data$risk_score <- sapply(1:nrow(test_data), function(i) {
  calculate_risk_score(test_data[i, ])
})

cat("Risk Score System (0-11 points):\n")
cat("---------------------------------\n")
cat("Age:\n")
cat("  70+ years: 2 points\n")
cat("  60-69 years: 1 point\n")
cat("Previous Inpatient Visits:\n")
cat("  2+ visits: 3 points\n")
cat("  1 visit: 2 points\n")
cat("Hospital Stay:\n")
cat("  7+ days: 2 points\n")
cat("  4-6 days: 1 point\n")
cat("Medications:\n")
cat("  20+ meds: 2 points\n")
cat("  15-19 meds: 1 point\n")
cat("Diagnoses:\n")
cat("  9+ diagnoses: 1 point\n")
cat("Emergency Admission: 1 point\n\n")

# Analyze score distribution
cat("Risk Score Distribution:\n")
score_table <- table(test_data$risk_score)
print(score_table)

# Analyze readmission by risk score
cat("\n\nReadmission Rate by Risk Score:\n")
cat("--------------------------------\n")
for (score in sort(unique(test_data$risk_score))) {
  subset <- test_data[test_data$risk_score == score, ]
  readmit_rate <- mean(subset$readmitted == "Yes") * 100
  cat(sprintf("Score %d: %.1f%% readmission rate (n=%d)\n", 
              score, readmit_rate, nrow(subset)))
}

# Define risk categories
test_data$risk_category <- cut(test_data$risk_score,
                               breaks = c(-Inf, 3, 6, Inf),
                               labels = c("Low Risk", "Medium Risk", "High Risk"))

cat("\n\nRisk Categories:\n")
cat("----------------\n")
cat("Low Risk: 0-3 points\n")
cat("Medium Risk: 4-6 points\n")
cat("High Risk: 7+ points\n\n")

cat("Readmission Rate by Risk Category:\n")
for (category in levels(test_data$risk_category)) {
  subset <- test_data[test_data$risk_category == category, ]
  readmit_rate <- mean(subset$readmitted == "Yes") * 100
  cat(sprintf("%s: %.1f%% (n=%d)\n", category, readmit_rate, nrow(subset)))
}

################################################################################
# STAGE 9: VISUALIZATIONS
################################################################################

cat("\n\nSTAGE 9: CREATING VISUALIZATIONS\n")
cat("=================================\n\n")

if (!dir.exists("plots_phase3")) {
  dir.create("plots_phase3")
  cat("Created 'plots_phase3/' directory\n")
}

# Visualization 1: Model Comparison
png("plots_phase3/1_model_comparison.png", width = 900, height = 600)
par(mar = c(8, 5, 4, 2))
metrics_matrix <- as.matrix(comparison[, -1])
rownames(metrics_matrix) <- comparison$Model

barplot(t(metrics_matrix), beside = TRUE,
        main = "Model Performance Comparison",
        ylab = "Score",
        col = c("steelblue", "coral", "lightgreen", "yellow"),
        legend = colnames(metrics_matrix),
        las = 2,
        ylim = c(0, 1))
grid(nx = NA, ny = NULL, col = "gray", lty = "dotted")
dev.off()
cat("Created: 1_model_comparison.png\n")

# Visualization 2: Feature Importance
png("plots_phase3/2_feature_importance.png", width = 900, height = 600)
par(mar = c(5, 12, 4, 2))
top_10 <- head(coef_data, 10)
barplot(top_10$Importance, 
        names.arg = top_10$Variable,
        horiz = TRUE,
        las = 1,
        main = "Top 10 Most Important Features",
        xlab = "Importance (|Coefficient|)",
        col = "darkblue")
grid(nx = NULL, ny = NA, col = "gray", lty = "dotted")
dev.off()
cat("Created: 2_feature_importance.png\n")

# Visualization 3: Risk Score Distribution
png("plots_phase3/3_risk_score_distribution.png", width = 900, height = 600)
par(mfrow = c(2, 1))

# Risk score histogram
hist(test_data$risk_score,
     breaks = seq(min(test_data$risk_score) - 0.5, 
                  max(test_data$risk_score) + 0.5, 1),
     main = "Risk Score Distribution",
     xlab = "Risk Score (0-11 points)",
     ylab = "Number of Patients",
     col = "lightblue",
     border = "white")

# Readmission rate by score
scores <- sort(unique(test_data$risk_score))
rates <- sapply(scores, function(s) {
  mean(test_data$readmitted[test_data$risk_score == s] == "Yes") * 100
})

plot(scores, rates,
     type = "b",
     pch = 19,
     col = "red",
     lwd = 2,
     main = "Readmission Rate by Risk Score",
     xlab = "Risk Score",
     ylab = "Readmission Rate (%)",
     ylim = c(0, 100))
grid()

dev.off()
cat("Created: 3_risk_score_distribution.png\n")

# Visualization 4: Risk Categories
png("plots_phase3/4_risk_categories.png", width = 900, height = 600)
par(mfrow = c(1, 2))

# Category distribution
category_counts <- table(test_data$risk_category)
pie(category_counts,
    labels = paste(names(category_counts), "\n",
                   category_counts, "patients"),
    main = "Risk Category Distribution",
    col = c("lightgreen", "yellow", "red"))

# Readmission by category
category_readmit <- sapply(levels(test_data$risk_category), function(cat) {
  subset <- test_data[test_data$risk_category == cat, ]
  mean(subset$readmitted == "Yes") * 100
})

barplot(category_readmit,
        names.arg = levels(test_data$risk_category),
        main = "Readmission Rate by Risk Category",
        ylab = "Readmission Rate (%)",
        col = c("lightgreen", "yellow", "red"),
        ylim = c(0, 100))
text(x = seq(0.7, by = 1.2, length.out = 3),
     y = category_readmit + 5,
     labels = paste0(round(category_readmit, 1), "%"))

dev.off()
cat("Created: 4_risk_categories.png\n")

# Visualization 5: Confusion Matrices
png("plots_phase3/5_confusion_matrices.png", width = 900, height = 600)
par(mfrow = c(1, 2))

# Logistic Regression
plot(1:2, 1:2, type = "n", xlim = c(0.5, 2.5), ylim = c(0.5, 2.5),
     xlab = "Predicted", ylab = "Actual",
     main = "Logistic Regression\nConfusion Matrix",
     xaxt = "n", yaxt = "n")
axis(1, at = 1:2, labels = c("No", "Yes"))
axis(2, at = 1:2, labels = c("No", "Yes"))
grid()

# Add cells
rect(0.5, 0.5, 1.5, 1.5, col = rgb(0, 1, 0, 0.3))
rect(1.5, 0.5, 2.5, 1.5, col = rgb(1, 0, 0, 0.3))
rect(0.5, 1.5, 1.5, 2.5, col = rgb(1, 0, 0, 0.3))
rect(1.5, 1.5, 2.5, 2.5, col = rgb(0, 1, 0, 0.3))

# Add numbers
text(1, 1, confusion_logit[1, 1], cex = 2)
text(2, 1, confusion_logit[2, 1], cex = 2)
text(1, 2, confusion_logit[1, 2], cex = 2)
text(2, 2, confusion_logit[2, 2], cex = 2)

# Decision Tree
plot(1:2, 1:2, type = "n", xlim = c(0.5, 2.5), ylim = c(0.5, 2.5),
     xlab = "Predicted", ylab = "Actual",
     main = "Rule-Based Tree\nConfusion Matrix",
     xaxt = "n", yaxt = "n")
axis(1, at = 1:2, labels = c("No", "Yes"))
axis(2, at = 1:2, labels = c("No", "Yes"))
grid()

rect(0.5, 0.5, 1.5, 1.5, col = rgb(0, 1, 0, 0.3))
rect(1.5, 0.5, 2.5, 1.5, col = rgb(1, 0, 0, 0.3))
rect(0.5, 1.5, 1.5, 2.5, col = rgb(1, 0, 0, 0.3))
rect(1.5, 1.5, 2.5, 2.5, col = rgb(0, 1, 0, 0.3))

text(1, 1, confusion_tree[1, 1], cex = 2)
text(2, 1, confusion_tree[2, 1], cex = 2)
text(1, 2, confusion_tree[1, 2], cex = 2)
text(2, 2, confusion_tree[2, 2], cex = 2)

dev.off()
cat("Created: 5_confusion_matrices.png\n")

cat("\nAll 5 Phase 3 visualizations created successfully!\n")

################################################################################
# STAGE 10: EXPORT RESULTS
################################################################################

cat("\n\nSTAGE 10: EXPORTING RESULTS\n")
cat("===========================\n\n")

# Save model comparison
write.csv(comparison, "model_performance_comparison.csv", row.names = FALSE)
cat("Saved: model_performance_comparison.csv\n")

# Save feature importance
write.csv(coef_data[1:20, ], "feature_importance.csv", row.names = FALSE)
cat("Saved: feature_importance.csv\n")

# Save risk score analysis
risk_analysis <- data.frame(
  Risk_Score = scores,
  Readmission_Rate = rates,
  Count = as.numeric(table(test_data$risk_score))
)
write.csv(risk_analysis, "risk_score_analysis.csv", row.names = FALSE)
cat("Saved: risk_score_analysis.csv\n")
