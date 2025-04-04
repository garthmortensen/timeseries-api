import yfinance as yf
import pandas as pd
import logging as l
from typing import Dict, List, Any

def fetch_market_data(symbols: List[str], start_date: str, end_date: str, interval: str = "1d") -> Dict[str, Dict[str, Any]]:
    """
    Fetch market data from Yahoo Finance.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data frequency (1d, 1wk, 1mo, etc.)
        
    Returns:
        Dict with dates as keys and nested dicts of symbol data as values
    """
    l.info(f"Fetching market data for {symbols} from {start_date} to {end_date}")
    
    # Format to match the structure returned by generate_data
    data_dict = {}
    
    try:
        # Download data for all symbols at once
        data = yf.download(symbols, start=start_date, end=end_date, interval=interval)
        
        # If only one symbol, yfinance returns a different structure
        if len(symbols) == 1:
            # Handle single symbol case
            data.columns = pd.MultiIndex.from_product([['Adj Close'], symbols])
        
        # Extract adjusted close prices
        prices = data['Adj Close']
        
        # Convert to the format expected by TimeSeriesDataResponse
        for date_idx, row in prices.iterrows():
            str_date = str(date_idx.date())
            data_dict[str_date] = row.to_dict()
            
        return data_dict
    except Exception as e:
        l.error(f"Error fetching data from Yahoo Finance: {e}")
        raise
