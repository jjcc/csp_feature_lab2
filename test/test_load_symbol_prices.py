#!/usr/bin/env python3

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the function to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a02merge_macro_features import _load_symbol_prices


class TestLoadSymbolPrices(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.symbol = "AAPL"
        self.start_date = "2025-06-06"
        self.end_date = "2025-08-08"
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_from_csv_file_with_Date_Close_columns(self):
        """Test loading symbol prices from CSV file with Date/Close columns"""
        path = "output/price_cache"
        symbol = "AAPL"
        
        result = _load_symbol_prices(symbol, path, self.start_date, self.end_date, use_yf=False)

        self.assertEqual(result.name, "Close")
        self.assertGreater(len(result), 0)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result.index))
        self.assertGreater(result.iloc[0], 0)
        self.assertGreater(result.iloc[-1], 0)
    
    
    def test_csv_file_not_exists(self):
        """Test when CSV file does not exist and use_yf=False"""
        result = _load_symbol_prices(self.symbol, self.temp_dir, self.start_date, self.end_date, use_yf=False)
        
        self.assertEqual(result.name, "Close")
        self.assertTrue(result.empty)
        self.assertEqual(result.dtype, float)
    
    def test_no_px_dir_provided(self):
        """Test when no px_dir is provided and use_yf=False"""
        result = _load_symbol_prices(self.symbol, None, self.start_date, self.end_date, use_yf=False)
        
        self.assertEqual(result.name, "Close")
        self.assertTrue(result.empty)
        self.assertEqual(result.dtype, float)
    
    def test_date_filtering(self):
        """Test that date filtering works correctly"""
        # Create test CSV with dates outside the range
        test_data = pd.DataFrame({
            'Date': ['2022-12-30', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-15'],
            'Close': [140.0, 150.0, 151.0, 149.0, 160.0]
        })
        csv_path = Path(self.temp_dir) / f"{self.symbol}.csv"
        test_data.to_csv(csv_path, index=False)
        
        result = _load_symbol_prices(self.symbol, self.temp_dir, "2023-01-01", "2023-01-03", use_yf=False)
        
        self.assertEqual(len(result), 3)  # Should only include dates in range
        self.assertAlmostEqual(result.iloc[0], 150.0)  # First should be 2023-01-01
        self.assertAlmostEqual(result.iloc[-1], 149.0)  # Last should be 2023-01-03
    
    def test_invalid_dates_handling(self):
        """Test handling of invalid dates in CSV"""
        test_data = pd.DataFrame({
            'Date': ['2023-01-01', 'invalid-date', '2023-01-03'],
            'Close': [150.0, 151.0, 149.0]
        })
        csv_path = Path(self.temp_dir) / f"{self.symbol}.csv"
        test_data.to_csv(csv_path, index=False)
        
        result = _load_symbol_prices(self.symbol, self.temp_dir, self.start_date, self.end_date, use_yf=False)
        
        # Should only include valid dates (invalid row should be dropped)
        self.assertEqual(len(result), 2)
    
    @patch('a02merge_macro_features.yf')
    def test_yfinance_fallback_success(self, mock_yf):
        """Test successful yfinance fallback when CSV not available"""
        # Mock yfinance download
        mock_data = pd.DataFrame({
            'Close': [150.0, 151.0, 149.0]
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_yf.download.return_value = mock_data
        
        result = _load_symbol_prices(self.symbol, None, self.start_date, self.end_date, use_yf=True)
        
        mock_yf.download.assert_called_once_with(self.symbol, start=self.start_date, end=self.end_date)
        self.assertEqual(result.name, "Close")
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result.iloc[0], 150.0)
    
    @patch('a02merge_macro_features.yf')
    def test_yfinance_fallback_exception(self, mock_yf):
        """Test yfinance fallback when it raises an exception"""
        mock_yf.download.side_effect = Exception("Network error")
        
        result = _load_symbol_prices(self.symbol, None, self.start_date, self.end_date, use_yf=True)
        
        self.assertEqual(result.name, "Close")
        self.assertTrue(result.empty)
        self.assertEqual(result.dtype, float)
    
    def test_empty_symbol(self):
        """Test with empty symbol string"""
        result = _load_symbol_prices("", None, self.start_date, self.end_date, use_yf=False)
        
        self.assertEqual(result.name, "Close")
        self.assertTrue(result.empty)
        self.assertEqual(result.dtype, float)


if __name__ == '__main__':
    unittest.main()