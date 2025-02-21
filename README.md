# Modeling-Project 

Below is a complete solution for creating an R Shiny app that allows users to explore and model datasets from the `ExamPAData` package. The app will enable users to select a dataset, choose an appropriate machine learning model (for regression or classification), and view performance metrics and visualizations. I'll provide the code for the two main Shiny files: `ui.R` and `server.R`, along with instructions on how to set up and run the app.

---

## Solution: R Shiny App Files

### `ui.R`
This file defines the user interface of the Shiny app.

```r
library(shiny)

fluidPage(
  titlePanel("ExamPAData Explorer: Predictive Analytics Tool"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("dataset", "Choose a dataset:", 
                  choices = c("customer_phone_calls", "patient_length_of_stay", "patient_num_labs", 
                              "actuary_salaries", "june_pa", "customer_value", "exam_pa_titanic", 
                              "apartment_apps", "health_insurance", "student_success", 
                              "readmission", "auto_claim", "boston", "bank_loans")),
      
      selectInput("model", "Choose a model:", choices = NULL)  # Dynamically updated based on dataset
    ),
    
    mainPanel(
      verbatimTextOutput("metrics"),
      plotOutput("plot")
    )
  )
)
```

### `server.R`
This file contains the server logic to handle user inputs, train models, and generate outputs.

```r
library(shiny)
library(ExamPAData)
library(caret)
library(dplyr)
library(ggplot2)

# Predefined list of datasets with problem type and target variable
dataset_info <- list(
  customer_phone_calls = list(type = "classification", target = "call"),
  patient_length_of_stay = list(type = "regression", target = "length_of_stay"),
  patient_num_labs = list(type = "regression", target = "num_labs"),
  actuary_salaries = list(type = "regression", target = "salary"),
  june_pa = list(type = "regression", target = "claim_amount"),
  customer_value = list(type = "regression", target = "value"),
  exam_pa_titanic = list(type = "classification", target = "survived"),
  apartment_apps = list(type = "classification", target = "approved"),
  health_insurance = list(type = "regression", target = "charges"),
  student_success = list(type = "classification", target = "success"),
  readmission = list(type = "classification", target = "readmitted"),
  auto_claim = list(type = "regression", target = "claim_amount"),
  boston = list(type = "regression", target = "price"),
  bank_loans = list(type = "classification", target = "default")
)

server <- function(input, output, session) {
  
  # Reactive expression to load the selected dataset
  selected_data <- reactive({
    data(list = input$dataset, package = "ExamPAData")
    get(input$dataset)
  })
  
  # Update model choices based on the problem type
  observe({
    problem_type <- dataset_info[[input$dataset]]$type
    if (problem_type == "regression") {
      updateSelectInput(session, "model", choices = c("Linear Regression", "Random Forest"))
    } else {
      updateSelectInput(session, "model", choices = c("Logistic Regression", "Random Forest"))
    }
  })
  
  # Reactive expression to train the model
  model_fit <- reactive({
    data <- selected_data()
    target <- dataset_info[[input$dataset]]$target
    problem_type <- dataset_info[[input$dataset]]$type
    
    # Split data into training and testing sets
    set.seed(123)
    train_index <- createDataPartition(data[[target]], p = 0.8, list = FALSE)
    train_data <- data[train_index, ]
    test_data <- data[-train_index, ]
    
    # Define formula
    formula <- as.formula(paste(target, "~ ."))
    
    # Train model based on selected model and problem type
    if (input$model == "Linear Regression" && problem_type == "regression") {
      train(formula, data = train_data, method = "lm")
    } else if (input$model == "Logistic Regression" && problem_type == "classification") {
      train(formula, data = train_data, method = "glm", family = "binomial")
    } else if (input$model == "Random Forest") {
      train(formula, data = train_data, method = "rf")
    }
  })
  
  # Calculate and display performance metrics
  output$metrics <- renderPrint({
    data <- selected_data()
    target <- dataset_info[[input$dataset]]$target
    problem_type <- dataset_info[[input$dataset]]$type
    
    # Split data into training and testing sets
    set.seed(123)
    train_index <- createDataPartition(data[[target]], p = 0.8, list = FALSE)
    test_data <- data[-train_index, ]
    
    # Predict on test data
    predictions <- predict(model_fit(), newdata = test_data)
    
    if (problem_type == "regression") {
      # Calculate RMSE and R-squared
      rmse <- sqrt(mean((test_data[[target]] - predictions)^2))
      r_squared <- cor(test_data[[target]], predictions)^2
      cat("RMSE:", rmse, "\nR-squared:", r_squared)
    } else {
      # Calculate accuracy
      if (is.factor(test_data[[target]])) {
        test_data[[target]] <- as.factor(test_data[[target]])
        predictions <- factor(predictions, levels = levels(test_data[[target]]))
      }
      confusion <- confusionMatrix(predictions, test_data[[target]])
      cat("Accuracy:", confusion$overall['Accuracy'])
    }
  })
  
  # Generate visualization
  output$plot <- renderPlot({
    data <- selected_data()
    target <- dataset_info[[input$dataset]]$target
    problem_type <- dataset_info[[input$dataset]]$type
    
    # Split data into training and testing sets
    set.seed(123)
    train_index <- createDataPartition(data[[target]], p = 0.8, list = FALSE)
    test_data <- data[-train_index, ]
    
    # Predict on test data
    predictions <- predict(model_fit(), newdata = test_data)
    
    if (problem_type == "regression") {
      # Scatter plot of actual vs predicted
      ggplot(test_data, aes(x = .data[[target]], y = predictions)) +
        geom_point() +
        geom_abline(slope = 1, intercept = 0, color = "red") +
        labs(x = "Actual", y = "Predicted", title = "Actual vs Predicted")
    } else {
      # Confusion matrix heatmap
      confusion <- confusionMatrix(predictions, test_data[[target]])
      ggplot(data = as.data.frame(confusion$table), aes(x = Reference, y = Prediction, fill = Freq)) +
        geom_tile() +
        scale_fill_gradient(low = "white", high = "blue") +
        labs(title = "Confusion Matrix")
    }
  })
}
```

---

## How to Set Up and Run the App

### 1. Install Required Packages
Before running the app, ensure you have the necessary R packages installed. Run the following command in your R console:

```r
install.packages(c("shiny", "ExamPAData", "caret", "dplyr", "ggplot2"))
```

### 2. Create the App Files
- **Create a new directory** (e.g., `ExamPAExplorer`) on your computer.
- **Save the files**:
  - Copy the `ui.R` code above into a file named `ui.R`.
  - Copy the `server.R` code above into a file named `server.R`.
  - Place both files in the `ExamPAExplorer` directory.

### 3. Run the App
- Open R or RStudio.
- Set your working directory to the folder containing the files:
  ```r
  setwd("path/to/ExamPAExplorer")
  ```
- Run the app:
  ```r
  shiny::runApp()
  ```
This will launch the Shiny app in your default web browser.

---

## App Functionality

### Features
- **Dataset Selection**: Choose from a list of datasets in the `ExamPAData` package via a dropdown menu.
- **Model Selection**: Select a machine learning model (e.g., Linear Regression, Logistic Regression, or Random Forest) from a dropdown menu that updates dynamically based on whether the dataset is for regression or classification.
- **Performance Metrics**: View metrics such as RMSE and R-squared for regression tasks, or accuracy for classification tasks.
- **Visualizations**: See a scatter plot of actual vs. predicted values for regression, or a confusion matrix heatmap for classification.

### How It Works
1. The app loads the selected dataset from `ExamPAData`.
2. Based on the dataset’s problem type (regression or classification), it offers appropriate model options.
3. It trains the chosen model on 80% of the data and tests it on the remaining 20%.
4. The app then calculates performance metrics and generates a plot to visualize the results.

---

## Notes
- The app assumes the `ExamPAData` package datasets are structured with a clear target variable, as defined in the `dataset_info` list.
- For simplicity, minimal preprocessing is included. In a production environment, you might want to add data cleaning or feature scaling steps.
- The Random Forest model works for both regression and classification, making it a versatile option.

This app is a valuable tool for exploring predictive analytics and practicing for the Society of Actuaries' Predictive Analytics Exam (Exam PA). Enjoy experimenting with the datasets and models!
