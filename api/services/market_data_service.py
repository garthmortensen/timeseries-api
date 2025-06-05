#!/usr/bin/env python3
# timeseries-api/api/services/market_data_service.py

import yfinance as yf
import pandas as pd
import logging as l
from typing import Dict, List, Any, Tuple

import pandas_datareader.data as web
from datetime import datetime


def fetch_market_data_yfinance(symbols: List[str], start_date: str, end_date: str, interval: str = "1d") -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Fetch market data from Yahoo Finance.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data frequency (1d, 1wk, 1mo, etc.)
        
    Returns:
        Tuple of:
        - List of dictionaries, each with date and symbol values (for API response)
        - DataFrame with datetime index (for internal library use)
    """
    l.info(f"Fetching market data for {symbols} from {start_date} to {end_date}")
    
    try:
        # Download data for all symbols at once
        data = yf.download(symbols, start=start_date, end=end_date, interval=interval)
        
        # Process data to get prices DataFrame similar to generate_price_series()
        if len(symbols) == 1:
            # For single symbol, ensure we have a DataFrame with Close price column
            if isinstance(data.columns, pd.MultiIndex):
                # Try Adj Close first, fall back to Close if not available
                if 'Adj Close' in data.columns.get_level_values(0):
                    prices = data['Adj Close']
                else:
                    prices = data['Close']
            else:
                # Not a multi-index, just get the column directly
                if 'Adj Close' in data.columns:
                    prices = data['Adj Close']
                else:
                    prices = data['Close']
            
            # Rename column to the symbol
            prices = prices.to_frame(name=symbols[0])
        else:
            # For multiple symbols, extract Adj Close or Close
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close']
            else:
                prices = data['Close']
        
        # Ensure the index is datetime 
        prices.index = pd.to_datetime(prices.index)
        
        # For API responses: Convert to list of dictionaries (records format)
        # Use lowercase 'date' for API consistency
        records = prices.reset_index().rename(columns={'index': 'date'}).to_dict('records')
        
        # Convert datetime objects to strings
        for record in records:
            if isinstance(record.get('date'), pd.Timestamp):
                record['date'] = record['date'].strftime('%Y-%m-%d')
        
        # Note: Keep prices DataFrame with datetime index for internal library use
        # DO NOT reset_index() on the prices DataFrame that will be used with the library

        return records, prices
    
    except Exception as e:
        l.error(f"Error fetching data from Yahoo Finance: {e}")
        raise


def fetch_market_data_stooq(symbols: List[str], start_date: str, end_date: str, interval: str = "1d") -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Fetch market data from Stooq.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data frequency (1d, 1wk, 1mo)
        
    Returns:
        Tuple of:
        - List of dictionaries, each with date and symbol values (for API response)
        - DataFrame with datetime index (for internal library use)
    """
    l.info(f"Fetching market data from Stooq for {symbols} from {start_date} to {end_date}")
    
    # Convert dates to datetime objects
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Prepare DataFrame to hold all data
    all_data = pd.DataFrame()
    
    try:
        # Fetch data for each symbol
        for symbol in symbols:
            try:
                # Get data from Stooq
                df = web.DataReader(symbol, 'stooq', start=start_dt, end=end_dt)
                
                # Stooq data typically has Open, High, Low, Close columns
                # Extract only the Close price and rename to symbol
                if 'Close' in df.columns:
                    symbol_data = df[['Close']].rename(columns={'Close': symbol})
                    
                    # Merge with main DataFrame
                    if all_data.empty:
                        all_data = symbol_data
                    else:
                        all_data = all_data.join(symbol_data, how='outer')
                else:
                    l.warning(f"No 'Close' column in data for {symbol} from Stooq")
            
            except Exception as e:
                l.error(f"Error fetching data for {symbol} from Stooq: {e}")
        
        # Sort by date
        all_data = all_data.sort_index()
        
        # Convert to list of dictionaries for API response
        records = all_data.reset_index().rename(columns={"Date": "date"}).to_dict("records")
        
        # Convert datetime objects to strings
        for record in records:
            if isinstance(record.get("date"), pd.Timestamp):
                record["date"] = record["date"].strftime("%Y-%m-%d")
        
        return records, all_data
    
    except Exception as e:
        l.error(f"Error fetching data from Stooq: {e}")
        raise