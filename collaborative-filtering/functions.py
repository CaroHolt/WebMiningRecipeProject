import numpy as np
import pandas as pd


def filter_user_item(df, filter_user, filter_recipes, filter_method_user, filter_method_recipes):

    # filter_methods = {"iqr": filter_by_iqr, "random_sampling": filter_by_random_sampling}
    filtered_df = df

    if filter_user:
        # filter_function_user = filter_methods.get(filter_method_user)
        ratings_per_user = filtered_df.groupby("user_id").rating.count()
        lower_range, upper_range = calculate_iqr(ratings_per_user)
        filtered_df = filtered_df.groupby("user_id").filter(lambda x: (len(x) > lower_range) & (len(x) < upper_range))

    if filter_recipes:
        ratings_per_user = filtered_df.groupby("recipe_id").rating.count()
        lower_range, upper_range = calculate_iqr(ratings_per_user)
        filtered_df = filtered_df.groupby("recipe_id").filter(lambda x: (len(x) > lower_range) & (len(x) < upper_range))

    return filtered_df


def apply_filter_method(df, column, filter_method):
    if filter_method == "iqr":
        ratings_per_user = df.groupby(column).rating.count()
        lower_range, upper_range = calculate_iqr(ratings_per_user)
        filtered_df = df.groupby(column).filter(lambda x: (len(x) > lower_range) & (len(x) < upper_range))
    return df


def filter_by_random_sampling(df):
    return df


def calculate_iqr(filter_column):
    sorted(filter_column)
    q1, q3 = np.percentile(filter_column, [25, 75])
    iqr = q3 - q1
    lower_range = q1 - (1.5 * iqr)
    upper_range = q3 + (1.5 * iqr)
    return lower_range, upper_range


def random_sample():
    return 0