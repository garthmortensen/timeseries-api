#!/usr/bin/env python3
# yfinance_test.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("Simple Yahoo Finance Data Test")
print("-" * 30)

# Test with reliable stock symbols
symbols = ["AAPL"]
print(f"Testing symbols: {symbols}")

# Date range: last 30 days
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
print(f"Date range: {start_date} to {end_date}")

try:
    # Fetch data
    print("\nFetching data...")
    data = yf.download(symbols, start=start_date, end=end_date, progress=False)
    
    # Check if we got any data
    if data.empty:
        print("ERROR: No data returned!")
    else:
        print(f"SUCCESS! Got {len(data)} days of data")
        
        # Show first few rows of closing prices
        print("\nFirst 3 days of closing prices:")
        print(data['Close'].head(3))
        
        # Show data shape and available columns
        print(f"\nData shape: {data.shape} (rows, columns)")
        print(f"Available price types: {list(data.columns.levels[0])}")
        
        # Test specific symbols one by one
        print("\nTesting individual symbols:")
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            info = ticker.history(period="1d")
            if not info.empty:
                print(f"  ✓ {symbol}: Data available")
            else:
                print(f"  ✗ {symbol}: No data")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")

print("\nTest complete!")
