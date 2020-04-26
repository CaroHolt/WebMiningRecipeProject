import selenium

from selenium import webdriver
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


