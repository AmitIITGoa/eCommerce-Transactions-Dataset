# eCommerce-Transactions-Dataset

# E2E Implementation of EDA, Lookalike Model & Clustering

This repository provides an end-to-end pipeline for:
1. **Exploratory Data Analysis (EDA)** on customer transaction data,  
2. **Lookalike Modeling** to find top-3 similar customers for any given customer,  
3. **Customer Segmentation (Clustering)** using K-Means.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Dependencies & Installation](#dependencies--installation)
- [Usage](#usage)
- [Project Steps](#project-steps)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Lookalike Model](#2-lookalike-model)
  - [3. Customer Segmentation / Clustering](#3-customer-segmentation--clustering)
- [Results](#results)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview
In many business scenarios, understanding customers’ behaviors, spending patterns, and similarities can help companies:
- Identify top revenue-generating segments.
- Personalize marketing campaigns based on “lookalike” or “similar” customers.
- Perform effective segmentation for tailored strategies.

This project demonstrates:
1. **EDA** to glean insights on revenue trends, product categories, and more.  
2. **Lookalike Modeling** leveraging feature engineering, encoding, and **cosine similarity** to recommend the top-3 similar customers for each user.  
3. **Clustering** to categorize customers into segments using the **Davies–Bouldin (DB) index** to select the optimal number of clusters.

---

## Data Description
Three sample CSV files:
1. **Customers.csv**  
   - Columns: `CustomerID, CustomerName, Region, SignupDate`, etc.
2. **Products.csv**  
   - Columns: `ProductID, ProductName, Category, Price`, etc.
3. **Transactions.csv**  
   - Columns: `TransactionID, CustomerID, ProductID, TransactionDate, Quantity, Price, TotalValue`, etc.

> **Note**: You may adapt the script to your own data format, but ensure the columns match or modify the code accordingly.

---

## Dependencies & Installation

- **Python 3.7+** (Recommended 3.9 or later)
- **pandas**
- **numpy**
- **matplotlib**
- **seaborn**
- **scikit-learn** (1.2+ recommended)
- **jupyter** (optional, for notebook exploration)

### Install via pip
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
