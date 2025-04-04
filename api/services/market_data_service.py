import yfinance as yf
import pandas as pd
import logging as l
from typing import Dict, List, Any, Tuple

def fetch_market_data(symbols: List[str], start_date: str, end_date: str, interval: str = "1d") -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Fetch market data from Yahoo Finance.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data frequency (1d, 1wk, 1mo, etc.)
        
    Returns:
        Tuple of:
        - Dict with dates as keys and nested dicts of symbol data as values
        - DataFrame with datetime index and columns for each symbol
    """
    l.info(f"Fetching market data for {symbols} from {start_date} to {end_date}")
    
    # Format to match the structure returned by generate_data()
    data_dict = {}
    
    try:
        # Download data for all symbols at once
        data = yf.download(symbols, start=start_date, end=end_date, interval=interval)
        
        # Ensure we have a MultiIndex or regular DataFrame with Adj Close
        if len(symbols) == 1:
            # For single symbol, ensure we have a DataFrame with Adj Close
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Adj Close']
            else:
                prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            
            # Rename column to the symbol
            prices = prices.to_frame(name=symbols[0])
        else:
            # For multiple symbols, extract Adj Close
            prices = data['Adj Close']
        
        # Ensure the index is datetime 
        prices.index = pd.to_datetime(prices.index)
        
        # Populate data_dict in the same structure as generate_price_series() found in py package
        for date_idx, row in prices.iterrows():
            str_date = str(date_idx)
            data_dict[str_date] = row.to_dict()
        
        return data_dict, prices
    
    except Exception as e:
        l.error(f"Error fetching data from Yahoo Finance: {e}")
        raise
