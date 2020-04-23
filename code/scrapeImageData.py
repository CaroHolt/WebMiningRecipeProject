import requests
import urllib.request
import time
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd

url = 'https://www.food.com/recipe/67256'

driver = webdriver.Chrome("C:/Webdrivers/chromedriver")
driver.get(url)

images = driver.find_elements_by_tag_name('img')
recipe_images = []
for image in images:
    if(image.get_attribute('class') == 'recipe-image__img'):
        recipe_images.append(image.get_attribute('src'))

