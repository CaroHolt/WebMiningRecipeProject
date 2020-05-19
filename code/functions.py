import selenium
import nltk

from selenium import webdriver
from rake_nltk import Rake
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import inflect
import re, string, unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import SnowballStemmer
from nltk.stem import LancasterStemmer, WordNetLemmatizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

# deal with NAs in our dataframe
def deal_with_NAs(df):
    df.drop(df[df["name"].isna()].index, inplace =True)
    df["description"].fillna("", inplace=True)
    df.loc[144074, "minutes"]= 25
    df.drop(df[df["name"]=="how to preserve a husband"].index, inplace=True)

# Calculate average ratings for each recipe
def get_avg_recipe_rating(interactions_df, recipes_df):
    #Average ratings
    num_interactions = interactions_df.groupby("recipe_id")["date"].count()
    #only consider the ratings (>0) into the mean, not the reviews w/o ratings
    mean_ratings = round(interactions_df[interactions_df["rating"]!=0].groupby("recipe_id")["rating"].mean(), 2)
    #merge
    df_rmerged = recipes_df.join(num_interactions, how="left", on="recipe_id").join(mean_ratings, how="left", on="recipe_id")
    df_rmerged = df_rmerged.rename(columns ={"date":"num_interactions", "rating":"avg_rating"})
    return df_rmerged

def filter_byquality(df):
    df.drop(df[(df["n_steps"]==1)&(df["num_interactions"]==1)].index, axis=0, inplace =True)
    print("Shape after removing 1 step recipes w/ low interactions:", df.shape)
    df.drop(df[df["avg_rating"].isna()].index, axis=0, inplace =True)
    print("Shape after removing recipes w/o ratings:", df.shape)
    df.drop(df[(df['minutes']==0)].index, axis=0, inplace=True)
    print('Shape after removing 0 minutes interaction w/ low interactions:', df.shape)

def filter_byinteractions(num_interactions, age, df, older):
    """
    older: boolean
    """
    if (older==True):
        index_remove= df[(df["num_interactions"]<=num_interactions) & (df["age"]>age)]["recipe_id"].index
        df.drop(index_remove, axis=0, inplace=True)
        print(f'Shape after filtering recipes with less than {num_interactions} interactions and older than {age} years old: {df.shape}')
    else:
        index_remove= df[(df["num_interactions"]<=num_interactions) & (df["age"]<=age)]["recipe_id"].index
        df.drop(index_remove, axis=0, inplace=True)
        print(f'Shape after filtering recipes with less than {num_interactions} interactions and younger than {age} years old: {df.shape}')
def choose_best(interactions, ratings, n_dupl):
    # number of interaction are different -> there exists a maximum
    if((len(interactions) != len(set(interactions)))) :
        return interactions.idxmax(axis=1)
    else:# return the maximum rating or any of the duplicate recipes
        return ratings.idxmax(axis=1)

def remove_duplicates(df):
    dupl_recipes = pd.DataFrame(df[df["name"].duplicated(keep=False)])
    dupl_rgrouped= dupl_recipes.groupby('name').groups
    
    to_keep = []
    
    for name in dupl_rgrouped:
        n_dupl = len(dupl_rgrouped[name])
        if(n_dupl == 2):
            index1=dupl_rgrouped[name][0]
            index2=dupl_rgrouped[name][1]
            
            interactions = dupl_recipes.loc[[index1, index2], ['num_interactions']].num_interactions
            ratings = dupl_recipes.loc[[index1, index2], ['avg_rating']].avg_rating
            to_keep.append(choose_best(interactions, ratings, n_dupl))
        elif (n_dupl==3):
            index1=dupl_rgrouped[name][0]
            index2=dupl_rgrouped[name][1]
            index3=dupl_rgrouped[name][2]
            
            interactions = dupl_recipes.loc[[index1, index2, index3], ['num_interactions']].num_interactions
            ratings = dupl_recipes.loc[[index1, index2, index3], ['avg_rating']].avg_rating
            to_keep.append(choose_best(interactions, ratings, n_dupl))
        else:
            print("Error")
            break
            
        df.drop(df.index.intersection(to_keep), axis=0, inplace=True)
    print('Shape after dropping duplicates:', df.shape)

#generate URL for every recipe
def generate_URL(df):
    df["URL"] = df.apply(lambda row: "https://www.food.com/recipe/"+" ".join(row["name"].split()).replace(" ", "-") 
                         +"-"+str(row["recipe_id"]), axis=1)
    print(f'URLs created for each of the {len(df.index)} recipes')
    return df

def get_rating_dist(df_column):
    ratings_series=pd.Series(df_column.value_counts())
    ratings_series.plot.bar()
    print(f'Percent of 5 star rating interactions: {round((ratings_series.loc[5]/len(df_column)*100),2)}%')
    print(f'Percent of 4 star rating interactions: {round((ratings_series.loc[4]/len(df_column)*100),2)}%')
    print(f'Percent of 3 star rating interactions: {round((ratings_series.loc[3]/len(df_column)*100),2)}%')
    print(f'Percent of 2 star rating interactions: {round((ratings_series.loc[2]/len(df_column)*100),2)}%')
    print(f'Percent of 1 star rating interactions: {round((ratings_series.loc[1]/len(df_column)*100),2)}%')
    
#source: https://github.com/nding17/YelpRecommendation/blob/master/notebooks/Content%20Based%20Models.ipynb
def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_special(words):
    """Remove special signs like &*"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[-,$()#+&*]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""  
    stopwords = nltk.corpus.stopwords.words('english')
    myStopWords = []
    stopwords.extend(myStopWords)
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words

def to_lowercase(words):
    """Convert words to lowercase"""
    new_words=[]
    for word in words:
        new_words.append(word.lower())
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    #stemmer = SnowballStemmer('english')
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize_lemmatize(words):
    words = remove_special(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    #words = stem_words(words)
    words = lemmatize_verbs(words)
    return words

# Preprocess content data
def get_processed(data):
    processed = pd.DataFrame(data=[],columns = ['recipe_id', 'content'])
    new_texts = []

    for i in range(0, len(data)):
        recipe_id = data['recipe_id'].iloc[i]
        words = nltk.word_tokenize(data['content'].iloc[i])
        text = ' '.join(normalize_lemmatize(words))
        dfnew = pd.DataFrame([[recipe_id, text]], columns=['recipe_id', 'content'])
        new_texts.append(text)
        processed = processed.append(dfnew,ignore_index = True)

    return processed

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


###  Metrics ###

def get_hits(rec_dict, test_set):
    """
    Function computes the number of hits per user.

    Parameters:
    --------
    rect_dict: dict
    Dictionary containing a list of recommended recipes per user

    test_set: pd.DataFrame
    Data frame containing the user id, recipe id and rating

    Returns
    --------
    n_hist_per_user: list
    Number of hits per user
    """
    n_hits_per_user = []
    for uid in rec_dict:
        merged_recipe_lists = []
        # Get ground truth relevant recipes from test set
        user_interactions = test_set.loc[test_set['user_id'] == uid, ['recipe_id', 'rating']]
        relevant_recipes = user_interactions.loc[user_interactions.rating >= 4.0, 'recipe_id'].tolist()
        # Get predicted relevant recipes and compute number of hits
        # Count the number of equal elements by merging the lists and counting the occurences then substract 1.
        merged_recipe_lists.extend(rec_dict[uid])
        merged_recipe_lists.extend(relevant_recipes)
        n_hits = np.sum(np.unique(merged_recipe_lists, return_counts=True)[1] - 1)
        # Store the number of hits per user
        n_hits_per_user.append(n_hits)

    return n_hits_per_user


def get_avg_precision(n_hits_per_user, recset_size_users):
    return np.mean(np.array(n_hits_per_user) / recset_size_users)


def get_avg_recall(n_hits_per_user, testset_size_users):
    return np.mean(np.array(n_hits_per_user) / testset_size_users)


def get_f_one(precision, recall):
    if(precision == 0) & (recall == 0):
        f_one = 0
    else:
        f_one = 2 * precision * recall / (precision + recall)
    return f_one


def catalog_coverage(rec_dict, num_total_recipes):
    recommended_recipes = set()

    # Get the set of recommended recipes
    for uid, user_ratings in rec_dict.items():
        recommended_recipes.update(user_ratings)

    # Compute catalog coverage
    return (len(recommended_recipes) / num_total_recipes)

def hitrate(n_hits_per_user):
    hitr = np.sum((np.array(n_hits_per_user) > 0) * 1) / len(n_hits_per_user)
    return hitr