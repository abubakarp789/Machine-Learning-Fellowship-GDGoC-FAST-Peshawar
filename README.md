# Machine Learning Fellowship GDGoC FAST Peshawar

<div align="center">
<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version"/>
<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
<img src="https://img.shields.io/badge/Status-Completed-brightgreen.svg" alt="Project Status"/>
</div>

## 📚 Overview

This repository contains the tasks and projects completed during the Machine Learning Fellowship program at GDGoC FAST Peshawar (2025). The curriculum provides a comprehensive learning journey through the machine learning development lifecycle, from data preprocessing to model deployment, through a series of practical tasks and a final capstone project.

## 📋 Table of Contents

- [Installation](#-installation)
- [Tasks Overview](#-tasks-overview)
  - [Task 1: Data Preprocessing](#task-1-data-preprocessing)
  - [Task 2: Exploratory Data Analysis](#task-2-exploratory-data-analysis)
  - [Task 3: Supervised ML Model Development](#task-3-supervised-ml-model-development)
  - [Task 4: Unsupervised ML Model Development](#task-4-unsupervised-ml-model-development)
- [Final Project: Network Traffic Anomaly Detection](#-final-project-network-traffic-anomaly-detection)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)

## 🔧 Installation

To set up this project on your local machine:

```bash
# Clone the repository
git clone https://github.com/abubakarp789/Machine-Learning-Fellowship-GDGoC-FAST-Peshawar.git

# Navigate to the project directory
cd Machine-Learning-Fellowship-GDGoC-FAST-Peshawar

# Install required dependencies
pip install -r requirements.txt
```

## 📘 Tasks Overview

### Task 1: Data Preprocessing

**Problem Statement 1:** Airline Passenger Data Preprocessing

This task focuses on creating a unified, clean dataset from multiple heterogeneous data sources with inconsistent formats:

- **Data Integration from Multiple Sources:**
  - Online Booking System (JSON) - Direct airline website and mobile app bookings
  - Third-Party Travel Agency (XML) - Bookings made through external agencies
  - Airport Check-In System (JSON) - Actual check-in records at the airport

- **Data Cleaning and Standardization:**
  - Handling missing values across different fields (ticket class, seat number, ticket price)
  - Normalizing inconsistent formats for:
    - Date and time information (12-hour vs. 24-hour formats)
    - Phone numbers with varying formats and delimiters
    - Passenger names stored in different field structures
  - Outlier detection and treatment in numeric fields

- **Record Reconciliation:**
  - Identifying and resolving duplicate records across systems
  - Intelligently merging records to create the most complete passenger profiles
  - Prioritizing actual values over placeholder values when duplicates exist

- **Data Transformation:**
  - Creating a standardized CSV output with consistent field formats
  - Implementing proper data type conversion for downstream analysis
  - Ensuring data integrity throughout the transformation process

**Problem Statement 2:** Statistical Measure Performance Analysis

The second component focuses on analyzing when different measures of central tendency (mean, median, mode) are most appropriate:

- **Income Distribution Dataset Analysis:**
  - Creation of skewed income distribution with realistic outliers
  - Comparative evaluation of mean, median, and mode
  - Analysis of how outliers affect each statistical measure
  - Real-world implications of choosing incorrect measures for policy decisions

- **Product Rating Dataset Analysis:**
  - Generation of discrete rating data with clear modal patterns
  - Analysis of which central tendency measure best represents customer sentiment
  - Visualization of rating distributions and central tendency measures
  - Business implications for product development and marketing

- **Temperature Dataset Analysis:**
  - Creation of normally distributed temperature data with seasonal variations
  - Assessment of which statistical measure provides the most representative value
  - Impact of minor weather outliers on different measures
  - Practical applications for weather forecasting and climate analysis

The implementation demonstrates advanced data preprocessing skills with pandas and NumPy, showcasing the ability to handle complex, real-world data integration challenges and make informed decisions about statistical analysis approaches.

### Task 2: Exploratory Data Analysis

**Problem Statement:** Real Estate Market Analysis for Zingat

This project performs comprehensive exploratory data analysis on Zingat's dataset of 225,738 property listings to extract actionable insights about the Turkish real estate market:

- **Initial Data Exploration:**
  - Comprehensive dataset inspection and structure analysis
  - Distribution analysis of listings across different property categories and regions
  - Identification of key variables and their relationships
  - Summary statistics generation for understanding central tendencies and variability

- **Data Cleaning & Preprocessing:**
  - Implementation of missing value handling strategies
  - Outlier detection and treatment using statistical methods
  - Standardization of inconsistent data entries
  - Documentation of data quality issues and resolution approaches

- **Data Transformation & Feature Engineering:**
  - Variable transformations to improve analytical utility
  - Implementation of encoding techniques for categorical variables
  - Application of normalization and scaling methods
  - Creation of derived features to enhance analysis capabilities

- **Exploratory Analysis:**
  - Multi-dimensional visualization to reveal spatial and temporal patterns
  - Price analysis across different property types and locations
  - Investigation of factors influencing property valuations
  - Correlation analysis between property features and prices
  - Segmentation of the real estate market by region and property characteristics

- **Business Insights & Recommendations:**
  - Identification of market opportunities in different regions
  - Analysis of pricing trends and anomalies
  - Development of actionable insights to improve platform user experience
  - Strategic recommendations for Zingat's market positioning

The analysis leverages pandas for data manipulation, matplotlib and seaborn for advanced visualizations, and statistical techniques to generate insights that can guide business decision-making for an online real estate marketplace.

### Task 3: Supervised ML Model Development

**Problem Statement:** Heart Disease Severity Prediction

This task focuses on developing a multi-class classification model to predict heart disease severity (0-3) using the UCI Heart Disease Dataset:

- **Data Preprocessing:**
  - Handling missing values in clinical attributes
  - Encoding categorical variables (chest pain type, ECG results, etc.)
  - Scaling numerical features for optimal model performance
  - Feature selection based on correlation analysis

- **Exploratory Data Analysis:**
  - Correlation analysis between clinical features and heart disease severity
  - Distribution visualization of key cardiovascular indicators
  - Statistical analysis of demographic and clinical attributes

- **Model Development:**
  - Implementation of multiple classification algorithms:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - AdaBoost
  - Ensemble modeling with Voting Classifier
  - Hyperparameter tuning using cross-validation

- **Model Evaluation:**
  - Comprehensive performance assessment using accuracy, precision, recall, and F1-score
  - Comparative analysis between individual models and ensemble approach
  - Selection of optimal model based on performance metrics

- **Model Deployment:**
  - Serialization of trained model with joblib
  - Development of an interactive Streamlit web application
  - User-friendly interface for entering patient data and obtaining predictions
  - Clear visualization of prediction results and risk assessment

The project demonstrates a complete machine learning pipeline from preprocessing to deployment for medical diagnostics, highlighting the importance of interpretable predictions in healthcare applications.

### Task 4: Unsupervised ML Model Development

**Problem Statement:** Clustering, Association Rules, and Evaluation Metrics

This task explores several key areas of unsupervised machine learning and evaluation concepts:

1. **Task 0: Clustering the Smiley Dataset**
   - Implementation of K-means clustering (k=4) on smiley-shaped data
   - Implementation of DBSCAN clustering with parameter experimentation
   - Comparative analysis of how the algorithms handle non-linear clusters
   - Visualization of clustering results and parameter effects

2. **Task 1: Association Rule Mining with FP-Growth**
   - Application of FP-Growth algorithm to Eid sweet item transactions
   - Analysis of frequent itemsets and association patterns
   - Strategic recommendations for product placement optimization
   - Experimentation with different minimum support thresholds

3. **Task 2: Evaluation Metrics Analysis**
   - Comprehensive analysis of classification metrics (precision, recall, accuracy, F1-score)
   - Investigation of metric trade-offs in different problem contexts
   - Real-world applications and use case scenarios
   - Decision guidance for metric selection based on problem requirements

The notebooks demonstrate practical applications of unsupervised learning techniques and provide insights into evaluation methodologies for machine learning models.

## 🚀 Final Project: Network Traffic Anomaly Detection

The capstone project implements a hybrid machine learning system for detecting anomalies in network traffic that may indicate security threats:

- **Data Preprocessing:** Comprehensive cleaning and preparation of network traffic data
- **Exploratory Analysis:** Visualization of traffic patterns and feature relationships
- **Feature Engineering:** Creation of relevant features for anomaly detection
- **Unsupervised Learning:** DBSCAN clustering to identify unusual patterns
- **Supervised Classification:** Random Forest model to classify traffic as normal or anomalous
- **Model Evaluation:** Comprehensive performance assessment using appropriate metrics
- **Interactive Dashboard:** Streamlit web application for real-time anomaly detection

The project demonstrates end-to-end machine learning system development with a focus on security applications, implementing both supervised and unsupervised techniques and deploying them in a user-friendly interface.

## 💻 Technologies Used

- **Python:** Primary programming language
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Deep Learning:** Implementation from scratch
- **Web Application:** Streamlit
- **Development Environment:** Jupyter Notebooks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request for improvements, bug fixes, or additional features. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
