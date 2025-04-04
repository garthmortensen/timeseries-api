import yfinance as yf

# Define the index symbols
symbols = ['^DJI', '^HSI']
data = yf.download(symbols, period='5d', interval='1d')
print(data['Close'])
