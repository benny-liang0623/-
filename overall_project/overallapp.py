''' crawling data and filtering stock'''
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup


options = Options()
options.add_argument("--disable-notifications")

chrome = webdriver.Chrome('./chromedriver', chrome_options=options)
chrome.get("https://www.wantgoo.com/stock/dividend-yield")

for x in range(1, 4):
    chrome.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    time.sleep(5)

soup = BeautifulSoup(chrome.page_source, 'html.parser')

# stock name
titles = soup.find_all('td', {
    'class': 'zw'})

titlelist = []

for title in titles:
    titlelist.append(title.text)


# stock prices
prices = soup.find_all('td', {
    "c-model-dazzle": "text:close,class:upDn"})

pricelist = []

for price in prices:
    if price.text == None:
        pricelist.append("none")
    else:
        pricelist.append(price.text)

# dividend yield
dividends = soup.find_all('td', {
    "c-model": r"`{nearby4QuartersYield}%`"})

dividendlist = []

for dividend in dividends:
    dividendlist.append(dividend.text)


# cashdividend
cashdividends = soup.find_all('td', {
    "c-model": "cashDividend"})

cashdividendlist = []

for cashdividend in cashdividends:
    cashdividendlist.append(cashdividend.text)

# stocknumber
numbers = soup.find_all('a', {
    "c-model-dazzle": "text:id,href:url"})

numberlist = []

for number in numbers:
    numberlist.append(number.text)

chrome.quit()

# list to dataframe
all_list = [titlelist, pricelist, dividendlist, cashdividendlist, numberlist]

df = pd.DataFrame(all_list).transpose()
df.columns = ["stockname", "price",
              "dividend-field", "cash-dividend", "number"]
print(df)
df.to_csv('results.csv')


money = int(input("你要投資多少錢"))


topstock = data.iloc[:10]
topstock["profit"] = money/topstock["price"] * topstock["cash-dividend"]

print("------------------------------------------------------------------")
print("推薦這幾間公司給你")
print(topstock[["stockname", "price", "profit", "number"]])

''' machine learning and stock predict'''

company = (input("你要選擇哪一間公司"))
