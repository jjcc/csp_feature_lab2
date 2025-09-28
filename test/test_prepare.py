import unittest
import joblib
import pandas as pd
from a00build_basic_dataset import ensure_cache_dir
from service.data_prepare import _save_cached_price_data
from daily_stock_update import preload_prices_with_cache_by_time
from service.utils import download_prices_batched, get_symbols_last_few_days
import os



class TestPrepare(unittest.TestCase):

    def test_collect_previous_5days_option(self):
        out_dir = os.getenv("CACHE_DIR", "./output")
        folder = "option/put"
        end_date_str = "2025-08-22"
        end_date = pd.to_datetime(end_date_str)

        files, symbols = get_symbols_last_few_days(folder, end_date)
        print(f"Symbols for last 5 trading days up to {end_date}: {symbols}")

        start_date = end_date - pd.Timedelta(days=90)
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        prices = {}

        cache_dir = ensure_cache_dir(out_dir)

        fetched = download_prices_batched(symbols, start_date, end_date, batch_size=40, threads=True)
        for s, price_df in fetched.items():
            prices[s] = price_df
            _save_cached_price_data(cache_dir, s, price_df)


        self.assertGreater(len(files), 0, "No files found for the specified dates")

    def test_get_cached_prices(self):
        out_dir = os.getenv("CACHE_DIR", "./output")
        #symbols = ["AAPL", "MSFT", "GOOGL"]


        folder = "option/put"
        end_date_str = "2025-08-22"
        end_date = pd.to_datetime(end_date_str)

        files, symbols = get_symbols_last_few_days(folder, end_date)


        previous_day, today = self.get_today_and_prevday()


        check_date = previous_day

        prices = preload_prices_with_cache_by_time(symbols, out_dir, check_date=check_date)

        for s in symbols:
            self.assertIn(s, prices)
            self.assertIsInstance(prices[s], pd.DataFrame)



    def test_update_latest_prices(self):
        from daily_stock_update import  stock_price_update
        stock_price_update(True)

        pass
    
    def test_get_vix(self):
        # First install: pip install vix-utils
        from vix_utils import get_vix_index_histories
        
        # Get current VIX cash data
        vix_data = get_vix_index_histories()
        current_vix = vix_data.iloc[-1]['Close']
        print(f"Current VIX: {current_vix:.2f}")
    
    def test_investigate_price_update(self):
        from daily_stock_update import  stock_price_update, preload_prices_with_cache_by_time
        out_dir = os.getenv("CACHE_DIR", "./output")
        cache_dir = ensure_cache_dir(out_dir)
        # get files  in cache_dir
        files = [f for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f))]
        #print(f"Cached price files: {files}")
        print(f"Cached price files count: {len(files)}")

        for f in files:
            if not f.endswith(".parquet"):
                continue
            path = os.path.join(cache_dir, f)
            df = pd.read_parquet(path)
            # the index column is datetime index, it's called "Date"
            min = df.index.min()
            max = df.index.max()
            print(f"{f}: Date range {min} to {max}, {len(df)} records")




        #folder = "option/put"
        #end_date_str = "2025-08-22"
        #end_date = pd.to_datetime(end_date_str)

        #files, symbols = get_symbols_last_few_days(folder, end_date)

        #previous_day, today = self.get_today_and_prevday()

        #check_date = previous_day

        #prices_before = preload_prices_with_cache_by_time(symbols, out_dir, check_date=check_date)

        #stock_price_update(True)

        #prices_after = preload_prices_with_cache_by_time(symbols, out_dir, check_date=check_date)

        #for s in symbols:
        #    df_before = prices_before.get(s)
        #    df_after = prices_after.get(s)
        #    if df_before is None or df_after is None:
        #        print(f"Symbol {s} missing in before or after prices.")
        #        continue
        #    if not df_before.equals(df_after):
        #        print(f"Prices for {s} have changed after update.")
        #    else:
        #        print(f"Prices for {s} are unchanged.")
    def test_get_config(self):
        from service.env_config import getenv
        data_dir = getenv("COMMON_DATA_DIR")
        basic_csv = getenv("COMMON_DATA_BASIC_CSV", "labeled_trades_normal.csv")
        output_dir = getenv("COMMON_OUTPUT_DIR", "./output")
        output_csv = getenv("COMMON_OUTPUT_CSV", "labeled_trades.csv")
        self.assertIsNotNone(data_dir)
        self.assertIsNotNone(basic_csv)
        self.assertIsNotNone(output_dir)
        self.assertIsNotNone(output_csv)
        print(f"Data dir: {data_dir}, Basic CSV: {basic_csv}")
        print(f"Output dir: {output_dir}, Output CSV: {output_csv}")

if __name__ == '__main__':
    unittest.main()
