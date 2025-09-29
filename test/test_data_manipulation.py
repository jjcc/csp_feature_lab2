import unittest
import pandas as pd



class TestDataManipulation(unittest.TestCase):

    def test_merge_tr_test1_dataframes(self):
        '''
        Merge original training dataset with the first test dataset
        '''
        file1 = "output/labeled_trades.csv"
        file2 = "output/labeled_trades_tr2.csv"
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df_merged = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        print(f"Merged DataFrame has {len(df_merged)} rows.")
        df_merged.to_csv("output/labeled_trades_tr_t1_merged.csv", index=False)

    def test_merge_tr_test1_dataframes_minus(self):
        '''
        Merge original training dataset with the first test dataset
        Remove the last N rows from the original training dataset before merging, to keep the size the same
        '''
        file1 = "output/labeled_trades.csv"
        file2 = "output/labeled_trades_tr2.csv"
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        length_test = len(df2)
        df1 = df1.iloc[:-length_test]
        df_merged = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        print(f"Merged DataFrame has {len(df_merged)} rows.")
        df_merged.to_csv("output/labeled_trades_tr_t1_merged_minus.csv", index=False)
    


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
