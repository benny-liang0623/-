'''crawler'''
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

data = pd.read_csv('results.csv')
money = int(input("你要投資多少錢"))


topstock = data.iloc[:10]
topstock["profit"] = money/topstock["price"] * topstock["cash-dividend"]

print("------------------------------------------------------------------")
print("推薦這幾間公司給你")
print(topstock[["stockname", "price", "profit", "number"]])




'''machine learning'''
import yfinance as yf
company = input("你要預測哪間公司")
comnum = topstock[topstock["stockname"] == "陽明"]["number"].astype("string")
comnum = [str(x) for x in comnum][0]
print(comnum)

import yfinance as yf

stocknum = "2609.TW"
stock = yf.Ticker(stocknum).history(period="10y")
stock = stock.filter(["Close"])
stock = stock.rename(columns={"Close":"GT"})



import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")
plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(stock["GT"],linewidth=1)
plt.show()


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_prices = scaler.fit_transform(stock.values)

#construct datasets
import numpy as np

moving_size = 60
all_x, all_y = [],[]
for i in range(len(scaled_prices)-moving_size):
  x = scaled_prices[i:i+moving_size]  
  y = scaled_prices[i+moving_size]
  all_x.append(x)
  all_y.append(y)

all_x, all_y = np.array(all_x), np.array(all_y)

#split data
DS_SPLIT = 0.8
train_ds_size = round(all_x.shape[0]*DS_SPLIT)
train_x, train_y = all_x[:train_ds_size], all_y[:train_ds_size]
test_x, test_y = all_x[train_ds_size:], all_y[train_ds_size:]

#construct model
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_x.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.summary()

model.compile(optimizer="adam", loss="mean_squared_error")

#train model
from tensorflow.keras.callbacks import EarlyStopping

callback = EarlyStopping(monitor="val_loss",patience=10, restore_best_weights = True)

model.fit(train_x,train_y,
      validation_split=0.2,
      callbacks=[callback],
      epochs=5)

#evalute_model
y_pred = model.predict(test_x)
#還原股價
preds = scaler.inverse_transform(y_pred)

#visualize result
train_df = df[:train_ds_size+moving_size]
test_df = df[train_ds_size+moving_size:]
test_df = test_df.assign(Predict=preds)

plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(train_df["GT"],linewidth=2)
plt.plot(test_df["GT"],linewidth=2)
plt.plot(test_df["Predict"],linewidth=1)
plt.legend(["Train","GT","Predict"])
plt.show()

