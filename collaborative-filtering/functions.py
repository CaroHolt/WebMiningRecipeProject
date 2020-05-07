import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

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
        random_sample = pd.concat([random_sample, sampled_observations], ignore_index=False)
        # In case the same observation was sampled twice for different columns
        random_sample.drop_duplicates(inplace=True)

    # Return the random sample
    return random_sample


def evaluation_plot(models, model_names, performance_measure="RMSE", plot_title=""):

    # Throw error message if the performance measure is not 'RMSE' or 'MAE'
    if performance_measure != "RMSE" and performance_measure != "MAE":
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
    plt.title(plot_title + "Performance", loc='center', fontsize=15)
    plt.plot(model_names, y, color='lightcoral', marker='o')
    plt.xlabel('Models', fontsize=15)
    plt.ylabel(performance_measure + "Value", fontsize=15)
    plt.grid(ls='dotted')
    return


# Evaluate filtering results
def evaluate_filtering(filtered_df, unfiltered_df):

    # Print the number of ratings included in the filtered dataset
    print(f"Number of ratings that are left: {len(filtered_df)}\n")

    # Print the number of users included in the filtered dataset
    print(f"Number of users that are left: {filtered_df.user_id.unique().size}\n")

    # Print the number of recipes included in the filtered dataset
    print(f"Number of recipes that are left: {filtered_df.recipe_id.unique().size}\n")

    # Print the fraction of ratings included in the filtered dataset
    print(f"Fraction of ratings that is left: {round(len(filtered_df) / len(unfiltered_df), 2)}\n")

    # Print the fraction of users included in the filtered dataset
    print(f"Fraction of users that is left: {round(filtered_df.user_id.unique().size / unfiltered_df.user_id.unique().size, 2)}\n")

    # Print the fraction of recipes included in the filtered dataset
    print(f"Fraction of recipes that is left: {round(filtered_df.recipe_id.unique().size / unfiltered_df.recipe_id.unique().size, 2)}\n")
    return

# Calculate novelty
def novelty(top_n, pop, u, n):
    """
    Computes the novelty for a list of recommendations
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    pop: dictionary
        A dictionary of all items alongside of its occurrences counter in the training data
        example: {1198: 893, 1270: 876, 593: 876, 2762: 867}
    u: integer
        The number of users in the training data
    n: integer
        The length of recommended lists per user
    Returns
    ----------
    novelty:
        The novelty of the recommendations in system level
    mean_self_information:
        The novelty of the recommendations in recommended top-N list level
    ----------
    Metric Defintion:
    Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R., & Zhang, Y. C. (2010).
    Solving the apparent diversity-accuracy dilemma of recommender systems.
    Proceedings of the National Academy of Sciences, 107(10), 4511-4515.
    """
    mean_self_information = []
    k = 0
    for uid, user_ratings in top_n.items():
        self_information = 0
        k += 1
        for i in user_ratings:
            self_information += np.sum(-np.log2(pop[i]/u))
    mean_self_information.append(self_information/n)
    novelty = sum(mean_self_information)/k
    return novelty, mean_self_information


# Personalization
# Relies on sorting!!!
def personalization(top_n):
    user_personalization = list()

    for uid_1, user_ratings_1 in top_n.items():
        for uid_2, user_ratings_2 in top_n.items():
            if uid_2 > uid_1:
                user_personalization.append(1- (len(set(user_ratings_2).intersection(set(user_ratings_1))) / 10))

    return (np.mean(user_personalization))


# Catalog coverage
def catalog_coverage(top_n, prediction_sample):
    # Build a counter object
    recommended_recipes = Counter()

    # Count the unique recommended recipes
    for uid, user_ratings in top_n.items():
        recommended_recipes.update(user_ratings)

    # Compute catalog coverage
    return len(recommended_recipes) / len(prediction_sample.recipe_id.unique())


def build_prediction_sample(ratings, sample_size):
    # Draw a sample from the recipes to evaluate the predicitions
    ratings_per_recipe = ratings.groupby("recipe_id").recipe_id.count().tolist()
    # Create a sample of 1000 recipes weight by their log frequency
    sampled_recipes = ratings.recipe_id.sort_values().drop_duplicates().sample(sample_size, weights=np.log(
        ratings_per_recipe)).values
    unique_users = ratings.user_id.drop_duplicates().values
    # Build all data frame that contains all possible combinations
    prediction_sample = pd.DataFrame("Na", columns=["rating"],
                                     index=pd.MultiIndex.from_product([unique_users, sampled_recipes])).sort_index()
    # Assign known ratings to the sample
    prediction_sample.rating = ratings.set_index(["user_id", "recipe_id"]).rating.sort_index()
    # Only keep those that are not rated yet
    prediction_sample = prediction_sample[prediction_sample.rating.isna()].reset_index(drop=False)
    prediction_sample.columns = ["user_id", "recipe_id", "rating"]

    return prediction_sample


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    # Just keep the recipe ids
    for uid, user_ratings in top_n.items():
        recommended_recipes = []
        for user_rating in user_ratings:
            recommended_recipes.append(user_rating[0])
        top_n[uid] = recommended_recipes

    return top_n