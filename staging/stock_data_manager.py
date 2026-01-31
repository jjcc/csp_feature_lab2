import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

class StockDataManager:
    def __init__(self, data_dir='stock_data', metadata_file='metadata.json'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_file = self.data_dir / metadata_file
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def download_batch(self, symbols, start_date='2024-04-01', end_date=None):
        """
        Downloads only what's needed for each symbol
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Categorize symbols
        new_symbols = []
        update_symbols = []
        
        for symbol in symbols:
            if symbol not in self.metadata:
                new_symbols.append(symbol)
            else:
                last_date = self.metadata[symbol]['end_date']
                if last_date < end_date:
                    update_symbols.append(symbol)
        
        print(f"New symbols: {len(new_symbols)}, Updates needed: {len(update_symbols)}")
        
        # Download new symbols (full history)
        if new_symbols:
            self._download_and_save(new_symbols, start_date, end_date, mode='new')
        
        # Download only incremental data for existing symbols
        if update_symbols:
            self._download_incremental(update_symbols, end_date)
        
        return len(new_symbols), len(update_symbols)
    
    def _download_and_save(self, symbols, start_date, end_date, mode='new'):
        """Download data and save to disk"""
        # Use yfinance batch download (much faster)
        data = yf.download(symbols, start=start_date, end=end_date, 
                          group_by='ticker', threads=True, progress=True)
        
        for symbol in symbols:
            try:
                if len(symbols) == 1:
                    df = data
                else:
                    df = data[symbol]
                
                if df.empty:
                    continue
                
                # Save to parquet (more efficient than CSV)
                file_path = self.data_dir / f"{symbol}.parquet"
                df.to_parquet(file_path)
                
                # Update metadata
                self.metadata[symbol] = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'last_updated': datetime.now().isoformat()
                }
            except Exception as e:
                print(f"Error saving {symbol}: {e}")
        
        self._save_metadata()
    
    def _download_incremental(self, symbols, end_date):
        """Download only missing dates for existing symbols"""
        for symbol in symbols:
            try:
                last_date = self.metadata[symbol]['end_date']
                # Start one day after last downloaded date
                start_date = (datetime.strptime(last_date, '%Y-%m-%d') + 
                            timedelta(days=1)).strftime('%Y-%m-%d')
                
                if start_date >= end_date:
                    continue  # Already up to date
                
                # Download incremental data
                new_data = yf.download(symbol, start=start_date, end=end_date, 
                                      progress=False)
                
                if new_data.empty:
                    continue
                
                # Load existing data and append
                file_path = self.data_dir / f"{symbol}.parquet"
                existing_data = pd.read_parquet(file_path)
                combined_data = pd.concat([existing_data, new_data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data.sort_index(inplace=True)
                
                # Save updated data
                combined_data.to_parquet(file_path)
                
                # Update metadata
                self.metadata[symbol]['end_date'] = end_date
                self.metadata[symbol]['last_updated'] = datetime.now().isoformat()
                
            except Exception as e:
                print(f"Error updating {symbol}: {e}")
        
        self._save_metadata()
    
    def get_data(self, symbol):
        """Retrieve data for a symbol"""
        file_path = self.data_dir / f"{symbol}.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        return None

# Usage example
#manager = StockDataManager()
#
## First batch
#batch1 = ['AAPL', 'MSFT', 'GOOGL'] * 200  # 600 symbols
#manager.download_batch(batch1, start_date='2024-04-01')
#
## Later update with overlapping symbols
#batch2 = ['AAPL', 'TSLA', 'NVDA'] * 200  # Some overlap
#manager.download_batch(batch2)  # Only downloads TSLA, NVDA fully; updates AAPL incrementally