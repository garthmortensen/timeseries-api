#!/usr/bin/env python3
# smoketest_stooq.py

import pandas as pd
import pandas_datareader.data as web
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
l = logging.getLogger(__name__)

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
                l.info(f"Fetching data for {symbol}...")
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

def run_smoketest():
    """
    Simple smoke test to verify the Stooq data fetcher is working.
    """
    # Use well-known symbols and recent dates
    symbols = ['AAPL', 'MSFT']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # 30 days of data
    
    print(f"\n===== STOOQ DATA FETCHER SMOKE TEST =====")
    print(f"Testing with symbols: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    print("=" * 40)
    
    try:
        # Call the function
        records, df = fetch_market_data_stooq(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Basic validation
        if len(records) > 0 and not df.empty:
            print(f"\n SUCCESS! Retrieved {len(records)} days of data")
            
            print("\n FIRST 3 ROWS OF DATA:")
            print(df.head(3))
            
            print(f"\n DATA SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
            
            print("\n COLUMNS:")
            print(df.columns.tolist())
            
            # Check for missing values
            missing = df.isna().sum()
            if missing.sum() > 0:
                print("\n  MISSING VALUES:")
                for col, count in missing.items():
                    if count > 0:
                        print(f"  - {col}: {count} missing")
            else:
                print("\n NO MISSING VALUES")
            
            # Basic stats
            print("\n BASIC STATS:")
            print(df.describe().round(2))
            
            return True
        else:
            print("\n ERROR: No data retrieved")
            return False
            
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    # Make sure pandas-datareader is installed
    try:
        import pandas_datareader
    except ImportError:
        print("pandas-datareader is not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "pandas-datareader"])
        print("pandas-datareader installed successfully.")
    
    # Run the smoke test
    success = run_smoketest()
    
    # Exit with appropriate status code
    import sys
    sys.exit(0 if success else 1)
