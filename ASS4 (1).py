import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

books_data = []
base = "https://books.toscrape.com/catalogue/page-{}.html"
page = 1
while True:
    url = base.format(page)
    r = requests.get(url)
    if r.status_code != 200:
        break
    soup = BeautifulSoup(r.text, "html.parser")
    items = soup.select(".product_pod")
    if not items:
        break
    for b in items:
        title = b.h3.a["title"]
        price = b.select_one(".price_color").text.strip()
        stock = b.select_one(".availability").text.strip()
        rating = b.p["class"][1]
        books_data.append([title, price, stock, rating])
    page += 1

df_books = pd.DataFrame(books_data, columns=["Title","Price","Availability","Rating"])
df_books.to_csv("books.csv", index=False)

chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)
driver.get("https://www.imdb.com/chart/top/")
time.sleep(3)

rows = driver.find_elements("css selector","table tbody tr")
imdb_data = []
for row in rows:
    rank = row.find_element("css selector",".ipc-title__text").text.split('.')[0]
    title = row.find_element("css selector","td.titleColumn a").text
    year = row.find_element("css selector","span.secondaryInfo").text.strip("()")
    rating = row.find_element("css selector","strong").text
    imdb_data.append([rank,title,year,rating])

driver.quit()

df_imdb = pd.DataFrame(imdb_data, columns=["Rank","Title","Year","Rating"])
df_imdb.to_csv("imdb_top250.csv", index=False)

weather_data = []
w = requests.get("https://www.timeanddate.com/weather/").text
s = BeautifulSoup(w, "html.parser")
table = s.select_one(".zebra.tb-wt.tb-hover")
rows = table.select("tbody tr")

for r in rows:
    city = r.select_one("td:nth-child(1) a").text
    tmp = r.select_one("td:nth-child(2)").text
    cond = r.select_one("td:nth-child(3)").text
    weather_data.append([city,tmp,cond])

df_weather = pd.DataFrame(weather_data, columns=["City","Temperature","Condition"])
df_weather.to_csv("weather.csv", index=False)
