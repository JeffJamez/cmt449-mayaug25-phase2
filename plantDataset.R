# ===============================
# PLANT DISEASE CLASSIFICATION ANALYSIS 
# By Wangeci Njiru,  Abayo Otieno & Kigotho James
# ===========================================

rm(list = ls())
setwd("/home/jeffjames/Documents/R/plantdatasets/Test2") 

library(EBImage)      
library(tidyverse)   
library(caret)        
library(e1071)       
library(rpart)        
library(rpart.plot)  
library(caTools)      
library(ggplot2)      
library(gridExtra)    
library(reshape2)    

# 1. DATA LOADING AND PREPROCESSING

# Define folder paths
healthy_path <- "Healthy"
powdery_path <- "Powdery" 
rust_path <- "Rust"

# Function to load and preprocess images
load_images <- function(folder_path, label) {
  # Get list of image files
  image_files <- list.files(folder_path, pattern = "\\.(jpg|jpeg|png|bmp)$", 
                            ignore.case = TRUE, full.names = TRUE)
  
  # Initialize lists to store data
  image_data <- list()
  labels <- c()
  
  cat("Loading", length(image_files), "images from", folder_path, "...\n")
  
  for (i in 1:length(image_files)) {
    tryCatch({
      # Read and preprocess image
      img <- readImage(image_files[i])
      
      # Convert to grayscale if colored
      if (length(dim(img)) == 3) {
        img <- channel(img, "gray")
      }
      
      # Resize image to standard size (64x64 for efficiency)
      img <- resize(img, w = 64, h = 64)
      
      # Store image data and label
      image_data[[i]] <- img
      labels[i] <- label
      
      # Progress indicator
      if (i %% 10 == 0) cat("Processed", i, "images\n")
      
    }, error = function(e) {
      cat("Error processing image", i, ":", e$message, "\n")
    })
  }
  
  return(list(images = image_data, labels = labels))
}

# Load images from all three categories
healthy_data <- load_images(healthy_path, "Healthy")
powdery_data <- load_images(powdery_path, "Powdery")
rust_data <- load_images(rust_path, "Rust")

# Combine all data
all_images <- c(healthy_data$images, powdery_data$images, rust_data$images)
all_labels <- c(healthy_data$labels, powdery_data$labels, rust_data$labels)

# Remove any NULL entries
valid_indices <- !sapply(all_images, is.null)
all_images <- all_images[valid_indices]
all_labels <- all_labels[valid_indices]

cat("Total images loaded:", length(all_images), "\n")
cat("Label distribution:\n")
print(table(all_labels))


# 2. FEATURE EXTRACTION

# Function to extract comprehensive features from images
extract_features <- function(img_list) {
  features <- data.frame()
  
  cat("Extracting features from", length(img_list), "images...\n")
  
  for (i in 1:length(img_list)) {
    img <- img_list[[i]]
    
    # Basic statistical features
    mean_intensity <- mean(img)
    std_intensity <- sd(as.vector(img))
    min_intensity <- min(img)
    max_intensity <- max(img)
    
    # Histogram features (intensity distribution)
    hist_data <- hist(as.vector(img), breaks = 10, plot = FALSE)
    hist_counts <- hist_data$counts / sum(hist_data$counts)  # Normalize
    
    # Texture features using local binary patterns approximation
    # Calculate gradient magnitude
    grad_x <- filter2(img, matrix(c(-1, 0, 1), nrow = 1))
    grad_y <- filter2(img, matrix(c(-1, 0, 1), ncol = 1))
    gradient_magnitude <- sqrt(grad_x^2 + grad_y^2)
    
    # Texture statistics
    texture_mean <- mean(gradient_magnitude)
    texture_std <- sd(as.vector(gradient_magnitude))
    
    # Edge density (high gradient areas)
    edge_density <- sum(gradient_magnitude > quantile(gradient_magnitude, 0.8)) / length(gradient_magnitude)
    
    # Symmetry features
    img_flip <- flip(img)
    symmetry_score <- cor(as.vector(img), as.vector(img_flip))
    
    # Combine all features
    feature_row <- data.frame(
      mean_intensity = mean_intensity,
      std_intensity = std_intensity,
      min_intensity = min_intensity,
      max_intensity = max_intensity,
      intensity_range = max_intensity - min_intensity,
      texture_mean = texture_mean,
      texture_std = texture_std,
      edge_density = edge_density,
      symmetry_score = symmetry_score,
      # Histogram bins
      hist_bin1 = hist_counts[1],
      hist_bin2 = hist_counts[2],
      hist_bin3 = hist_counts[3],
      hist_bin4 = hist_counts[4],
      hist_bin5 = hist_counts[5],
      hist_bin6 = hist_counts[6],
      hist_bin7 = hist_counts[7],
      hist_bin8 = hist_counts[8],
      hist_bin9 = hist_counts[9],
      hist_bin10 = hist_counts[10]
    )
    
    features <- rbind(features, feature_row)
    
    if (i %% 20 == 0) cat("Extracted features for", i, "images\n")
  }
  
  return(features)
}

# Extract features
features_df <- extract_features(all_images)
features_df$label <- factor(all_labels)

# Display feature summary
cat("\n=== FEATURE EXTRACTION SUMMARY ===\n")
cat("Total features extracted:", ncol(features_df) - 1, "\n")
cat("Dataset dimensions:", nrow(features_df), "x", ncol(features_df), "\n")
print(summary(features_df[, 1:9]))  # Show first 9 features

# 3. EXPLORATORY DATA ANALYSIS (EDA)

cat("\n=== EXPLORATORY DATA ANALYSIS ===\n")

# Basic statistics by class
cat("Class distribution:\n")
class_counts <- table(features_df$label)
print(class_counts)
cat("Percentages:\n")
print(round(prop.table(class_counts) * 100, 2))

# Statistical summary by class
cat("\nMean feature values by class:\n")
feature_means <- features_df %>%
  group_by(label) %>%
  summarise(
    mean_intensity = round(mean(mean_intensity), 4),
    std_intensity = round(mean(std_intensity), 4),
    texture_mean = round(mean(texture_mean), 4),
    edge_density = round(mean(edge_density), 4),
    .groups = 'drop'
  )
print(feature_means)

# Create comprehensive visualizations
cat("\nGenerating visualizations...\n")

# 1. Class distribution
p1 <- ggplot(features_df, aes(x = label, fill = label)) +
  geom_bar(alpha = 0.7) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  labs(title = "Distribution of Plant Disease Classes",
       x = "Disease Class", y = "Number of Images") +
  theme_minimal() +
  scale_fill_manual(values = c("Healthy" = "green", "Powdery" = "orange", "Rust" = "red"))

# 2. Mean intensity comparison
p2 <- ggplot(features_df, aes(x = label, y = mean_intensity, fill = label)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  labs(title = "Mean Intensity Distribution by Disease Class",
       x = "Disease Class", y = "Mean Intensity") +
  theme_minimal() +
  scale_fill_manual(values = c("Healthy" = "green", "Powdery" = "orange", "Rust" = "red"))

# 3. Texture features comparison
p3 <- ggplot(features_df, aes(x = label, y = texture_mean, fill = label)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Texture Features by Disease Class",
       x = "Disease Class", y = "Texture Mean") +
  theme_minimal() +
  scale_fill_manual(values = c("Healthy" = "green", "Powdery" = "orange", "Rust" = "red"))

# 4. Edge density comparison
p4 <- ggplot(features_df, aes(x = label, y = edge_density, fill = label)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Edge Density by Disease Class",
       x = "Disease Class", y = "Edge Density") +
  theme_minimal() +
  scale_fill_manual(values = c("Healthy" = "green", "Powdery" = "orange", "Rust" = "red"))

# 5. Scatter plot of key features
p5 <- ggplot(features_df, aes(x = mean_intensity, y = texture_mean, color = label)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(title = "Feature Relationship: Intensity vs Texture",
       x = "Mean Intensity", y = "Texture Mean") +
  theme_minimal() +
  scale_color_manual(values = c("Healthy" = "green", "Powdery" = "orange", "Rust" = "red"))

# 6. Correlation heatmap of selected features
selected_features <- features_df[, c("mean_intensity", "std_intensity", "texture_mean", 
                                     "edge_density", "intensity_range")]
correlation_matrix <- cor(selected_features)

# Convert correlation matrix to long format for plotting
cor_long <- reshape2::melt(correlation_matrix)
p6 <- ggplot(cor_long, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 2)), color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Feature Correlation Matrix", x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot 1: Class Distribution
print(p1)
cat("Press Enter to continue to next plot...")
readline()

# Plot 2: Mean Intensity Comparison
print(p2)
cat("Press Enter to continue to next plot...")
readline()

# Plot 3: Texture Features Comparison
print(p3)
cat("Press Enter to continue to next plot...")
readline()

# Plot 4: Edge Density Comparison
print(p4)
cat("Press Enter to continue to next plot...")
readline()

# Plot 5: Feature Relationship Scatter Plot
print(p5)
cat("Press Enter to continue to next plot...")
readline()

# Plot 6: Correlation Heatmap
print(p6)
cat("Press Enter to continue to next plot...")
readline()

# ADDITIONAL ADVANCED VISUALIZATIONS

# 7. Simple histogram comparison showing how image brightness varies
cat("Generating histogram comparison for image brightness...\n")

# This histogram shows How bright or dark the leaf images are for each disease type

p7 <- ggplot(features_df, aes(x = mean_intensity, fill = label)) +
  geom_histogram(bins = 15, alpha = 0.7, position = "identity") +
  facet_wrap(~label, ncol = 1) +
  labs(title = "How Bright Are the Leaf Images?",
       subtitle = "Histograms show the distribution of image brightness for each disease type",
       x = "Image Brightness (0 = Dark, 1 = Bright)", 
       y = "Number of Images") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("Healthy" = "green", "Powdery" = "orange", "Rust" = "red"))

 
cat("\nPLOT 7:\n")
cat("This graph shows how bright or dark the leaf images are.\n")
cat("KEY INSIGHT: Different diseases may make leaves appear brighter or darker.\n")

print(p7)
cat("Press Enter to continue to next plot...")
readline()

# 8. Simple bar chart showing average values for key features
cat("Generating simple bar chart comparison...\n")

# shows an Average measurements for different leaf characteristics

feature_averages <- features_df %>%
  group_by(label) %>%
  summarise(
    `Average Brightness` = round(mean(mean_intensity), 3),
    `Texture Roughness` = round(mean(texture_mean), 3),
    `Edge Sharpness` = round(mean(edge_density), 3),
    .groups = 'drop'
  ) %>%
  gather(key = "Measurement", value = "Average_Value", -label)

p8 <- ggplot(feature_averages, aes(x = label, y = Average_Value, fill = label)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = Average_Value), vjust = -0.3, size = 3) +
  facet_wrap(~Measurement, scales = "free_y", ncol = 3) +
  labs(title = "Average Leaf Characteristics by Disease Type",
       subtitle = "Bar charts comparing key measurements across healthy and diseased leaves",
       x = "Disease Type", y = "Average Measurement Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none",
        strip.text = element_text(size = 11, face = "bold")) +
  scale_fill_manual(values = c("Healthy" = "green", "Powdery" = "orange", "Rust" = "red"))

   
cat("\nPLOT 8 :\n")
cat("SHOWS: Average measurements for brightness, texture, and edge sharpness.\n")
cat("KEY INSIGHT: Each disease type has different average characteristics that help identify it.\n")

print(p8)
cat("Press Enter to continue to next plot...")
readline()

# 9. Simple scatter plot showing how two key measurements relate
cat("Performing simple relationship analysis...\n")

# shows: How image brightness relates to texture roughness
# Different colors show different disease types to see if they cluster together

p9 <- ggplot(features_df, aes(x = mean_intensity, y = texture_mean, color = label)) +
  geom_point(size = 3, alpha = 0.8) +
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed", size = 1) +
  labs(title = "How Image Brightness Relates to Texture Roughness",
       subtitle = "Each dot is one leaf image - different colors show different disease types",
       x = "Image Brightness (higher = brighter leaves)",
       y = "Texture Roughness (higher = more textured surface)",
       color = "Disease Type") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_manual(values = c("Healthy" = "green", "Powdery" = "orange", "Rust" = "red"))

   
cat("\n PLOT 9 EXPLANATION:\n")
cat("SHOWS: How image brightness and texture roughness are related.\n")
cat("KEY INSIGHT: Different diseases cluster in different areas, showing distinct patterns.\n")

print(p9)
cat("Press Enter to continue to next plot...")
readline()

# 10. Simple line chart showing how measurements change across disease types
cat("Generating line chart for disease progression patterns...\n")

# shows: How different measurements change from healthy to diseased leaves

measurement_trends <- features_df %>%
  group_by(label) %>%
  summarise(
    `Brightness` = mean(mean_intensity),
    `Texture` = mean(texture_mean),
    `Edge_Definition` = mean(edge_density),
    `Contrast` = mean(intensity_range),
    .groups = 'drop'
  ) %>%
  # Normalize values to 0-1 scale for comparison
  mutate(
    Brightness = (Brightness - min(Brightness)) / (max(Brightness) - min(Brightness)),
    Texture = (Texture - min(Texture)) / (max(Texture) - min(Texture)),
    Edge_Definition = (Edge_Definition - min(Edge_Definition)) / (max(Edge_Definition) - min(Edge_Definition)),
    Contrast = (Contrast - min(Contrast)) / (max(Contrast) - min(Contrast))
  ) %>%
  gather(key = "Measurement", value = "Normalized_Value", -label)

# Create ordered factor for logical disease progression
measurement_trends$label <- factor(measurement_trends$label, 
                                   levels = c("Healthy", "Powdery", "Rust"))

p10 <- ggplot(measurement_trends, aes(x = label, y = Normalized_Value, color = Measurement, group = Measurement)) +
  geom_line(size = 2, alpha = 0.8) +
  geom_point(size = 4, alpha = 0.9) +
  labs(title = "How Leaf Characteristics Change from Healthy to Diseased",
       subtitle = "Line chart showing measurement trends across disease progression",
       x = "Disease Progression", 
       y = "Measurement Level (0 = Lowest, 1 = Highest)",
       color = "Leaf Characteristic") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_brewer(type = "qual", palette = "Set2")

 
cat("\nPLOT 10 :\n")
cat("SHOWS: How leaf characteristics change as disease gets worse.\n")
cat("KEY INSIGHT: Each line shows how one characteristic changes from healthy to diseased leaves.\n")

print(p10)
cat("Press Enter to continue to next plot...")
readline()

# 11. Simple bar chart showing which measurements are most useful
cat("Generating feature usefulness ranking...\n")

#shows: Which leaf measurements are most helpful for identifying diseases

feature_importance <- data.frame(Feature = character(), F_statistic = numeric(), stringsAsFactors = FALSE)

feature_cols <- colnames(features_df)[1:(ncol(features_df)-1)]
for (feature in feature_cols) {
  formula_str <- paste(feature, "~ label")
  anova_result <- aov(as.formula(formula_str), data = features_df)
  f_stat <- summary(anova_result)[[1]]["label", "F value"]
  feature_importance <- rbind(feature_importance, 
                              data.frame(Feature = feature, F_statistic = f_stat))
}

# Sort by importance and create readable names
feature_importance <- feature_importance[order(feature_importance$F_statistic, decreasing = TRUE), ]

# Create more readable feature names
readable_names <- c(
  "mean_intensity" = "Image Brightness",
  "std_intensity" = "Brightness Variation",
  "texture_mean" = "Texture Roughness",
  "edge_density" = "Edge Sharpness",
  "intensity_range" = "Contrast Level",
  "min_intensity" = "Darkest Pixel",
  "max_intensity" = "Brightest Pixel",
  "symmetry_score" = "Leaf Symmetry"
)

# Apply readable names where available
feature_importance$Readable_Name <- ifelse(
  feature_importance$Feature %in% names(readable_names),
  readable_names[feature_importance$Feature],
  as.character(feature_importance$Feature)
)

# Take top 8 most important features
top_features <- head(feature_importance, 8)
top_features$Readable_Name <- factor(top_features$Readable_Name, levels = rev(top_features$Readable_Name))

p11 <- ggplot(top_features, aes(x = Readable_Name, y = F_statistic, fill = F_statistic)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = round(F_statistic, 1)), hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Which Leaf Measurements Are Most Useful for Disease Detection?",
       subtitle = "Bar chart ranking the most helpful characteristics (higher = more useful)",
       x = "Leaf Characteristic", y = "Usefulness Score",
       fill = "Usefulness\nScore") +
  theme_minimal() +
  theme(legend.position = "right")

 
cat("\n PLOT 11 EXPLANATION:\n")
cat("SHOWS: Which leaf measurements are best for identifying diseases.\n")
cat("KEY INSIGHT: Image brightness and texture are the most useful measurements.\n")

print(p11)
cat("Press Enter to continue to next plot...")
readline()

# 12. Simple histograms showing measurement distributions
cat("Generating simple distribution histograms...\n")

# shows: How measurements are spread out for the most important characteristics

# Get top 4 features from previous analysis
top_4_features <- head(top_features$Feature, 4)
top_4_readable <- head(top_features$Readable_Name, 4)

# Create histogram data
histogram_data <- features_df %>%
  select(label, all_of(as.character(top_4_features))) 

# Rename columns to readable names
names(histogram_data)[2:5] <- as.character(top_4_readable)

# Reshape for plotting
histogram_long <- histogram_data %>%
  gather(key = "Measurement", value = "Value", -label)

p12 <- ggplot(histogram_long, aes(x = Value, fill = label)) +
  geom_histogram(bins = 12, alpha = 0.7, position = "identity") +
  facet_grid(label ~ Measurement, scales = "free_x") +
  labs(title = "Distribution of the Most Important Leaf Measurements",
       subtitle = "Histograms showing how measurement values are spread for each disease type",
       x = "Measurement Value", y = "Number of Images",
       fill = "Disease Type") +
  theme_minimal() +
  theme(legend.position = "bottom",
        strip.text = element_text(size = 9)) +
  scale_fill_manual(values = c("Healthy" = "green", "Powdery" = "orange", "Rust" = "red"))

cat("\n PLOT 12:\n")
cat(": How the most important measurements are distributed across disease types.\n")
cat("KEY INSIGHT: Different diseases show different patterns in their measurement distributions.\n")

print(p12)
cat("Press Enter to continue to next plot...")
readline()


cat("\n=== BATCH DISPLAY OPTIONS ===\n")
# ==============================================================================
# 4. STATISTICAL ANALYSIS
# ==============================================================================

cat("\n=== STATISTICAL ANALYSIS ===\n")

# ANOVA tests for feature significance
cat("ANOVA Results for Key Features:\n")
features_to_test <- c("mean_intensity", "std_intensity", "texture_mean", "edge_density")

for (feature in features_to_test) {
  formula_str <- paste(feature, "~ label")
  anova_result <- aov(as.formula(formula_str), data = features_df)
  p_value <- summary(anova_result)[[1]]["label", "Pr(>F)"]
  cat(sprintf("%s: F-statistic = %.3f, p-value = %.6f %s\n", 
              feature, 
              summary(anova_result)[[1]]["label", "F value"],
              p_value,
              ifelse(p_value < 0.001, "***", ifelse(p_value < 0.01, "**", ifelse(p_value < 0.05, "*", "")))))
}

# Tukey's HSD for post-hoc analysis
cat("\nTukey's HSD Post-hoc Analysis for Mean Intensity:\n")
tukey_result <- TukeyHSD(aov(mean_intensity ~ label, data = features_df))
print(tukey_result)

# ==============================================================================
# 5. MACHINE LEARNING MODEL BUILDING
# ==============================================================================

cat("\n=== MACHINE LEARNING MODEL BUILDING ===\n")

# Prepare data for modeling
set.seed(123)  # For reproducibility

# Split data into training and testing sets (70-30 split)
sample_indices <- sample.split(features_df$label, SplitRatio = 0.7)
train_data <- features_df[sample_indices, ]
test_data <- features_df[!sample_indices, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")
cat("Training set class distribution:\n")
print(table(train_data$label))

# Remove label column for feature scaling
train_features <- train_data[, -ncol(train_data)]
test_features <- test_data[, -ncol(test_data)]
train_labels <- train_data$label
test_labels <- test_data$label

# Scale features
preprocess_params <- preProcess(train_features, method = c("center", "scale"))
train_features_scaled <- predict(preprocess_params, train_features)
test_features_scaled <- predict(preprocess_params, test_features)

# 6. MODEL TRAINING AND EVALUATION

# Model 1: Logistic Regression (Multinomial)
cat("\n--- Training Logistic Regression Model ---\n")
logistic_model <- train(
  x = train_features_scaled,
  y = train_labels,
  method = "multinom",
  trControl = trainControl(method = "cv", number = 5),
  trace = FALSE
)

# Predictions
logistic_pred <- predict(logistic_model, test_features_scaled)
logistic_accuracy <- confusionMatrix(logistic_pred, test_labels)$overall['Accuracy']

# Model 2: Decision Tree
cat("\n--- Training Decision Tree Model ---\n")
tree_model <- rpart(
  label ~ ., 
  data = train_data, 
  method = "class",
  control = rpart.control(cp = 0.01, minsplit = 10)
)

tree_pred <- predict(tree_model, test_data, type = "class")
tree_accuracy <- confusionMatrix(tree_pred, test_labels)$overall['Accuracy']

# Model 3: Support Vector Machine
cat("\n--- Training SVM Model ---\n")
svm_model <- svm(
  x = train_features_scaled,
  y = train_labels,
  kernel = "radial",
  cost = 1,
  gamma = 0.1
)

svm_pred <- predict(svm_model, test_features_scaled)
svm_accuracy <- confusionMatrix(svm_pred, test_labels)$overall['Accuracy']

# 7. MODEL EVALUATION AND COMPARISON

cat("\n=== MODEL PERFORMANCE COMPARISON ===\n")

# Create performance summary
model_performance <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Support Vector Machine"),
  Accuracy = c(logistic_accuracy, tree_accuracy, svm_accuracy),
  Accuracy_Percent = c(logistic_accuracy * 100, tree_accuracy * 100, svm_accuracy * 100)
)

print(model_performance)

# Detailed confusion matrices
cat("\n--- Logistic Regression Confusion Matrix ---\n")
logistic_cm <- confusionMatrix(logistic_pred, test_labels)
print(logistic_cm)

cat("\n--- Decision Tree Confusion Matrix ---\n")
tree_cm <- confusionMatrix(tree_pred, test_labels)
print(tree_cm)

cat("\n--- SVM Confusion Matrix ---\n")
svm_cm <- confusionMatrix(svm_pred, test_labels)
print(svm_cm)

# Feature importance for decision tree
cat("\n--- Decision Tree Feature Importance ---\n")
importance_scores <- tree_model$variable.importance
if (length(importance_scores) > 0) {
  importance_df <- data.frame(
    Feature = names(importance_scores),
    Importance = importance_scores
  ) %>% arrange(desc(Importance))
  print(head(importance_df, 10))
}

# 8. ADVANCED VISUALIZATIONS

cat("\n=== GENERATING ADVANCED VISUALIZATIONS ===\n")

# Model performance comparison
perf_plot <- ggplot(model_performance, aes(x = Model, y = Accuracy_Percent, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  geom_text(aes(label = paste0(round(Accuracy_Percent, 1), "%")), 
            vjust = -0.5, size = 4, fontweight = "bold") +
  labs(title = "Model Performance Comparison",
       x = "Machine Learning Model", y = "Accuracy (%)") +
  theme_minimal() +
  theme(legend.position = "none") +
  ylim(0, 100)

print(perf_plot)

# Decision tree visualization
cat("Generating decision tree plot...\n")
rpart.plot(tree_model, 
           main = "Decision Tree for Plant Disease Classification",
           type = 4, 
           extra = 2,
           fallen.leaves = TRUE,
           cex = 0.8)

# Confusion matrix heatmaps
create_cm_heatmap <- function(cm_table, title) {
  cm_df <- as.data.frame(as.table(cm_table))
  names(cm_df) <- c("Predicted", "Actual", "Frequency")
  
  ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Frequency)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Frequency), color = "black", size = 4) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = paste("Confusion Matrix:", title)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Best model confusion matrix heatmap
best_model_name <- model_performance$Model[which.max(model_performance$Accuracy)]
if (best_model_name == "Logistic Regression") {
  best_cm <- logistic_cm$table
} else if (best_model_name == "Decision Tree") {
  best_cm <- tree_cm$table
} else {
  best_cm <- svm_cm$table
}

cm_heatmap <- create_cm_heatmap(best_cm, best_model_name)
print(cm_heatmap)

# 9. FINAL SUMMARY AND INSIGHTS

cat("\n" + paste(rep("=", 80), collapse = "") + "\n")
cat("FINAL ANALYSIS SUMMARY\n")
cat(paste(rep("=", 80), collapse = "") + "\n")

cat("\n DATASET OVERVIEW:\n")
cat("• Total images analyzed:", length(all_images), "\n")
cat("• Classes: Healthy (", sum(all_labels == "Healthy"), "), ",
    "Powdery (", sum(all_labels == "Powdery"), "), ",
    "Rust (", sum(all_labels == "Rust"), ")\n", sep = "")
cat("• Features extracted per image:", ncol(features_df) - 1, "\n")

cat("\n KEY FINDINGS:\n")
cat("• Best performing model:", best_model_name, 
    "with", round(max(model_performance$Accuracy) * 100, 2), "% accuracy\n")

# Statistical significance
significant_features <- c()
for (feature in features_to_test) {
  formula_str <- paste(feature, "~ label")
  anova_result <- aov(as.formula(formula_str), data = features_df)
  p_value <- summary(anova_result)[[1]]["label", "Pr(>F)"]
  if (p_value < 0.05) {
    significant_features <- c(significant_features, feature)
  }
}

cat("• Statistically significant features (p < 0.05):", length(significant_features), "out of", length(features_to_test), "\n")
cat("• Most discriminative features:", paste(significant_features, collapse = ", "), "\n")

# Class separability insights
healthy_mean_intensity <- mean(features_df$mean_intensity[features_df$label == "Healthy"])
powdery_mean_intensity <- mean(features_df$mean_intensity[features_df$label == "Powdery"])
rust_mean_intensity <- mean(features_df$mean_intensity[features_df$label == "Rust"])

cat("• Mean intensity patterns:\n")
cat("  - Healthy leaves:", round(healthy_mean_intensity, 4), "\n")
cat("  - Powdery mildew:", round(powdery_mean_intensity, 4), "\n")
cat("  - Rust disease:", round(rust_mean_intensity, 4), "\n")

cat("\n RECOMMENDATIONS:\n")
cat("• Model recommendation:", best_model_name, "for deployment\n")
cat("• Key diagnostic features: Mean intensity, texture, and edge density\n")
cat("• Dataset quality: Good class balance with", length(all_images), "total samples\n")

cat("\n CLASSIFICATION PERFORMANCE:\n")
for (i in 1:nrow(model_performance)) {
  cat("• ", model_performance$Model[i], ": ", 
      round(model_performance$Accuracy_Percent[i], 2), "%\n", sep = "")
}

cat("\n ANALYSIS COMPLETE!\n")
cat("All models trained and evaluated successfully.\n")
cat("Visualizations and statistical tests completed.\n")
cat(paste(rep("=", 80), collapse = "") + "\n")

# Save results for presentation
cat("\nSaving results for presentation...\n")
write.csv(features_df, "plant_disease_features.csv", row.names = FALSE)
write.csv(model_performance, "model_performance_summary.csv", row.names = FALSE)

# Save model objects for future use
saveRDS(list(
  logistic_model = logistic_model,
  tree_model = tree_model,
  svm_model = svm_model,
  preprocess_params = preprocess_params
), "trained_models.rds")
