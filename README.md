# Actuary-Project: French Motor Third-Party Liability Analysis

## Overview
This repository contains a Python-based data analysis and predictive modeling pipeline for analyzing French motor third-party liability insurance claims. The project focuses on exploratory data analysis (EDA), feature binning, interaction term analysis, and fitting Poisson Generalized Linear Models (GLMs) to evaluate risk factors influencing claim frequency.

---

## Project Structure

* **`main.py`**: The primary execution script that coordinates data loading, visual exploratory analysis, interaction heatmaps, and model fitting routines.
* **`Plots.py`**: A dedicated visualization module containing functions to generate and save pair plots, percentile-binned exposure/frequency bar charts, and interaction plots.
* **`glmBuild.py`**: Model construction module utilizing `statsmodels` to fit Poisson Generalized Linear Models with log-exposure offsets.
* **`Interactions.py`**: Handles feature engineering for interaction terms, overdispersion analysis, and 2D heatmap generation.
* **`quickNotes.py`**: Scratchpad script used for testing code snippets and quick calculations.
