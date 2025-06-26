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
        - List of dictionaries, each with date and OHLCV values (for API response)
        - DataFrame with datetime index and OHLCV columns (for internal library use)
    """
    l.info(f"Fetching market data for {symbols} from {start_date} to {end_date}")
    
    try:
        # Download data for all symbols at once
        data = yf.download(symbols, start=start_date, end=end_date, interval=interval)
        
        # Process data to get full OHLCV DataFrame
        if len(symbols) == 1:
            # For single symbol, ensure we have a DataFrame with OHLCV columns
            if isinstance(data.columns, pd.MultiIndex):
                # Extract all OHLCV data
                ohlcv_data = pd.DataFrame()
                for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                    if col in data.columns.get_level_values(0):
                        ohlcv_data[f"{symbols[0]}_{col}"] = data[col].iloc[:, 0]
            else:
                # Rename columns to include symbol prefix
                ohlcv_data = data.copy()
                ohlcv_data.columns = [f"{symbols[0]}_{col}" for col in ohlcv_data.columns]
        else:
            # For multiple symbols, restructure the multi-index columns
            ohlcv_data = pd.DataFrame()
            for symbol in symbols:
                for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                    if col in data.columns.get_level_values(0) and symbol in data.columns.get_level_values(1):
                        ohlcv_data[f"{symbol}_{col}"] = data[col][symbol]
        
        # Ensure the index is datetime 
        ohlcv_data.index = pd.to_datetime(ohlcv_data.index)
        
        # For API responses: Convert to list of dictionaries with proper OHLCV structure
        records = []
        for idx, row in ohlcv_data.iterrows():
            record = {'date': idx.strftime('%Y-%m-%d')}
            
            # Group data by symbol
            for symbol in symbols:
                symbol_data = {}
                for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                    col_name = f"{symbol}_{col}"
                    if col_name in row and pd.notna(row[col_name]):
                        # Map to standard field names
                        field_map = {
                            'Open': 'open', 'High': 'high', 'Low': 'low', 
                            'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
                        }
                        symbol_data[field_map[col]] = float(row[col_name]) if col != 'Volume' else int(row[col_name])
                
                # Add symbol data to record
                record[symbol] = symbol_data
            
            records.append(record)

        return records, ohlcv_data
    
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