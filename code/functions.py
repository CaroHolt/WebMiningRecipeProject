import selenium
import nltk

from selenium import webdriver
from rake_nltk import Rake
import pandas as pd

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

