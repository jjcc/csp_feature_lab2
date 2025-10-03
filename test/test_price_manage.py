import os
import unittest
import joblib
import pandas as pd
from service.preprocess import add_dte_and_normalized_returns
from service.utils import prep_tail_training_df, fill_features_with_training_medians
from service.utils import ALL_FEATS
from train_tail_with_gex import _fill_features

class TestPriceData(unittest.TestCase):

    def setUp(self):
        pass

    def test_grouping(self):
        # Load the data
        file = "data/price_cache_dates.csv" 
        df = pd.read_csv(file)
        #df['end_date'] = pd.to_datetime(df['end_date'])
        groups = df.groupby('end_date')
        # Check that the groups are formed correctly
        for end_date, group in groups:
            print(f"End Date: {end_date}, Symbols: {group['symbol'].tolist()}")
        self.assertTrue(len(groups) > 0)

    def test_locate_date(self):
        # Load the data
        file = "data/price_cache_dates.csv" 
        df = pd.read_csv(file)
        symbol = "AAPL"
        # Locate the row for the symbol
        end_date = self.get_enddate(df, symbol)
        print(f"Symbol: {symbol}, End Date: {end_date}")
        self.assertIsNotNone(end_date)

    def get_enddate(self, df, symbol):
        row = df[df['symbol'] == symbol]
        # get the end date
        if not row.empty:
            end_date = row['end_date'].values[0]
        return end_date
    
    def test_grouped_update(self):
        from service.stock_data_manater2 import GroupedStockUpdater
        log_file = "data/price_cache_dates.csv"
        data_folder ="output/price_cache"
        # Create an instance of GroupedStockUpdater
        updater = GroupedStockUpdater(data_dir=data_folder, log_file=log_file)
        # Define a list of symbols to update
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        # Perform the batch update
        updater.update_batch(symbols)
        # Check that the log file has been updated
        self.assertTrue(os.path.exists('test_stock_log.csv'))
        log_df = pd.read_csv('test_stock_log.csv')
        print(log_df)
        self.assertTrue(not log_df.empty)

if __name__ == '__main__':
    unittest.main()
