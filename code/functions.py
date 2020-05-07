import selenium
import nltk

from selenium import webdriver
from rake_nltk import Rake
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

# deal with NAs in our dataframe
def deal_with_NAs(df):
    df.drop(df[df["name"].isna()].index, inplace =True)
    df["description"].fillna("", inplace=True)
    df.loc[144074, "minutes"]= 25
    df.drop(df[df["name"]=="how to preserve a husband"].index, inplace=True)


# Insert list of recipes to recommend
def get_image_source_url(recipe_id):
    url = 'https://www.food.com/recipe/'
    recipe_image_url = ""
    recipe_id = str(recipe_id)

    recipe_url = url + recipe_id

    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome("C:/Webdrivers/chromedriver",options=options)
    try:
        driver.get(recipe_url)
        images = driver.find_elements_by_tag_name('img')
        image_urls = []

        for image in images:
            if (image.get_attribute('class') == 'recipe-image__img'):
                image_urls.append(image.get_attribute('data-src'))
        recipe_image_url = image_urls[0]

    except Exception as err:
        print("No page found for {} - {}".format(recipe_id, err))

    return recipe_image_url

# Insert dataframe and name of old and new column as string
def get_keywords(df, oldColumnName, newColumnName):
    df[newColumnName] = ""
    for index, row in df.iterrows():
        oldEntries = row[oldColumnName]
        if oldColumnName == 'steps':
            oldEntries = row[oldColumnName].replace('[', '').replace(', ', '').replace(']', '').replace('and', '\'').split("\'")
            oldEntries = list(filter(None, oldEntries))
            all_entries = ""
            for i in oldEntries:
                all_entries += i
            oldEntries = all_entries
        r = Rake()
        r.extract_keywords_from_text(oldEntries)
        key_words_dict_scores = r.get_word_degrees()
        listEntries = list(key_words_dict_scores.keys())
        df.at[index, newColumnName] = ' '.join(listEntries)
    df.drop(columns=[oldColumnName], inplace=True)
    return df

# Compute cosine similarity matrix
def get_cos_sim_matrix(processed):
    tfidf = TfidfVectorizer(stop_words='english')
    processed['content'] = processed['content'].fillna('')
    tfidf_matrix = tfidf.fit_transform(processed['content'])
    svd = TruncatedSVD(n_components=10, random_state=42)
    tfidf_truncated = svd.fit_transform(tfidf_matrix) 
    cosine_sim = cosine_similarity(tfidf_truncated,tfidf_truncated)
    return cosine_sim

# preprocess interactions
def get_interaction_processed(processed, interactions):
  interactions_processed = interactions.loc[interactions.recipe_id.isin(processed.recipe_id)]\
                           .reset_index()\
                           .drop(columns=['index'])
  return interactions_processed

