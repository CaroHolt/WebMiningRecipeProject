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

# Insert list of recipes to recommend
def get_image_source_url(recipeList):
    url = 'https://www.food.com/recipe/'

    for index, row in recipeList.iterrows():
        recipe_id = str(row['id'])
        recipe_url = url + recipe_id

        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        driver = webdriver.Chrome("C:/Webdrivers/chromedriver",options=options)
        try:
            driver.get(recipe_url)
            images = driver.find_elements_by_tag_name('img')
            recipe_images = []
            for image in images:
                if (image.get_attribute('class') == 'recipe-image__img'):
                    recipe_images.append(image.get_attribute('src'))

            recipeList.at[index, 'ImageURL'] = recipe_images[0]
        except Exception as err:
            print("No page found for {} - {}".format(recipe_id, err))

    return recipeList

#Insert dataframe and name of old and new column as string
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

# Calculate recommendations from cosine similarity
def get_recommendation_cos(processed, interactions, recipe_id, user_id, cosine_sim, k):
    df_sim = pd.DataFrame(cosine_sim, index=processed['recipe_id'], columns=processed['recipe_id'])
    #adjust to sparse matrix
    user_rids = interactions[interactions['user_id']==user_id]['recipe_id'].values
    df_row = df_sim.loc[df_sim.index==recipe_id]
    df_user = df_row[user_rids]
    results = list(zip(list(df_user.columns), list(df_user.values[0])))
    results_ordered = sorted(results, key=lambda x: x[1], reverse=True)
    results_topk = np.array(results_ordered[1:k+1])
    return results_topk[:,0]

# Make the prediction of a rating
def predict_rating(interactions, user_id, recipe_ids):
    scores = []
    for rid in recipe_ids:
        score = interactions.loc[(interactions.user_id==user_id) & (interactions.recipe_id==rid)]['rating']
        scores.append(score.values[0])
        return np.mean(scores)

# Retriebe results of cosine similarity
def get_results_cos(processed, interactions, recipe, recipe_id, user_id, cosine_sim,k):
    actual = interactions.loc[(interactions.user_id==user_id) & (interactions.recipe_id==recipe_id)]['rating'].values[0]
    recipe_ids = get_recommendation_cos(processed,interactions,recipe_id,user_id,cosine_sim,k)
    prediction = predict_rating(interactions, user_id, recipe_ids)
    return actual, prediction

# Calculate coverage of predicted recipes
def get_coverage(processed, interactions, recipe, cosine_sim,k):
    interactions_processed = get_interaction_processed(processed, interactions)
    uid_sample = interactions_processed['user_id'].values
    rid_sample = interactions_processed['recipe_id'].values

    all_rids = interactions_processed['recipe_id'].unique()
    pred_rids = []

    for i in range(len(interactions_processed)):
        try:
          recipe_ids = get_recommendation_cos(processed,
                                                interactions_processed,
                                                rid_sample[i],
                                                uid_sample[i],
                                                cosine_sim,
                                                k)
          pred_rids += list(recipe_ids)
        except:
          next
    pred_bids = np.array(list(set(pred_rids)))
    return len(pred_bids)/len(all_rids)