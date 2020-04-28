import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne, CoClustering


def apply_filter(df, columns, filter_method, sample_size=10000):

    # Throw error message if the columns do not exist in the data frame
    if not pd.Series(columns).isin(df.columns).all():
        print("The columns do not exist in the data frame.")
        return

    # Specify the filter functions
    filter_functions = {"iqr_filtering": iqr_filter,
                        "random_sampling": random_sampling_filter}

    # Fetch the selected filter function
    filter_function = filter_functions.get(filter_method)

    # Apply the filter function
    df = filter_function(df, columns, kwargs_sample_size=sample_size)

    # Return the filtered data frame
    return df


def iqr_filter(df, columns, *args, **kwargs):

    # Initialize filtered data frame
    filtered_df = df

    # Apply the filter function to each column
    for column in columns:

        # Count the ratings column
        ratings_count = df.groupby(column).rating.count()

        # Calculate lower and upper range from IQR
        sorted(ratings_count)
        q1, q3 = np.percentile(ratings_count, [25, 75])
        iqr = q3 - q1
        lower_range = q1 - (1.5 * iqr)
        upper_range = q3 + (1.5 * iqr)

        # Filter data set
        inside_iqr = df.groupby(column).filter(lambda x: (len(x) > lower_range) & (len(x) < upper_range))[column]
        filtered_df = filtered_df[filtered_df[column].isin(inside_iqr)]

    # Return the filtered data frame
    return filtered_df


def random_sampling_filter(df, columns, *args, **kwargs):

    # Throw error message if the sample size is not passed
    if "kwargs_sample_size" not in kwargs:
        print("No sample size determined.")
        return

    # Throw errors message if the sample size is below zero
    if kwargs.get("kwargs_sample_size") < 0:
        print("Sample size below zero.")
        return

    # Initialize the empty sample data frame
    random_sample = pd.DataFrame([], columns=df.columns)

    # TODO: Split sample 50:50

    # TODO: Test ob ohne unique() funktioniert? Können user doppelt gesamplet werden?

    # Random sample from each selected column
    for column in columns:
        unique_column_values = df[column].unique()
        sampled_column_values = np.random.choice(unique_column_values, kwargs.get("kwargs_sample_size"), replace=False)
        sampled_observations = df.loc[df[column].isin(sampled_column_values)]
        random_sample = pd.concat([random_sample, sampled_observations], ignore_index=True)

    # Return the random sample
    return random_sample


def evaluation_plot(models, performance_measure):

    # TODO: Add model-names from Loop

    # Throw error message if the performance measure is not 'RMSE' or 'MAE'
    if measure != "RMSE" or measure != "MAE":
        print("Unknown performance measure.")
        return

    # Throw error message if the performance measure is not 'RMSE' or 'MAE'
    if not models:
        print("No models passed.")
        return

    # Define dictionary of performance measures
    performance_measures = {"RMSE": "test_rmse",
                            "MAE": "test_mae"}

    # Extract performance measure
    y = [round(model[performance_measures.get(performance_measure)].mean(), 4) for model in models]

    # Generate plot
    plt.figure(figsize=(18, 5))
    plt.title(paste(performance_measure, "Performance"), loc='center', fontsize=15)
    plt.plot(x, y, color='lightcoral', marker='o')
    plt.xlabel('Models', fontsize=15)
    plt.ylabel(paste(performance_measure, "Value"), fontsize=15)
    plt.grid(ls='dotted')
    plot.show()
    return

