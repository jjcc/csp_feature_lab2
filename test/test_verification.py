import unittest
import joblib
import pandas as pd
from service.utils import prep_tail_training_df, fill_features_with_training_medians
from service.utils import ALL_FEATS
from train_tail_with_gex import _fill_features

class TestVerification(unittest.TestCase):

    def setUp(self):
        pass

    def test_verify_real_return_calculation(self):
        """
        """
        # 'Unnamed: 0', 'symbol', 'baseSymbol', 'baseSymbolType', 'underlyingLastPrice', 'expirationDate', 
        # 'daysToExpiration', 'strike', 'moneyness', 'bidPrice', 'breakEvenBid', 'percentToBreakEvenBid', 'volume', 'openInterest', 'impliedVolatilityRank1y', 
        # 'delta', 'potentialReturn', 'potentialReturnAnnual', 'breakEvenProbability', 'expirationType', 'symbolType', 'tradeTime',
        #  '__source_file', 'bid', 'ask', 'expiry_close', 'win', 'entry_credit', 'exit_intrinsic', 'capital', 'total_pnl', 'return_pct', 'trade_date', 'symbol_norm', 'gex_total', 'gex_total_abs', 'gex_pos', 'gex_neg', 'gex_center_abs_strike', 'gex_flip_strike', 'gex_gamma_at_ul', 'gex_distance_to_flip', 'gex_sign_at_ul', 'gex_file', 'gex_missing', 'log1p_DTE', 'VIX', 'prev_close', 'ret_2d', 'ret_5d', 'ret_2d_norm', 'ret_5d_norm', 'prev_close_minus_strike', 'prev_close_minus_strike_pct', 'tail_proba', 'is_tail_preda'
        file = "output/tails_score/scored_with_tail_pct_mon_cut5.csv"
        col_to_keep = ["symbol", "baseSymbol", "underlyingLastPrice", "expirationDate", "daysToExpiration", "bidPrice",
                       "strike", 'potentialReturn','potentialReturnAnnual','tradeTime','expiry_close','entry_credit','exit_intrinsic',
                       'capital', 'total_pnl','return_pct']
        df = pd.read_csv(file)
        #cols = list(df.columns)
        #print("Columns in the CSV file:", cols)
        df = df[col_to_keep]
        output_file = "output/tails_score/scored_with_tail_pct_mon_cut5_short.csv"
        df.to_csv(output_file, index=False)
        # Check if both functions produce the same result
        #pd.testing.assert_frame_equal(result1, result2)

        

        #
        #assert(medians_x == medians_t)

    def test_drop_bidask(self):
        file = "output/labeled_trades_normal_gex_macro2.csv"
        df = pd.read_csv(file)
        df = df.drop(columns=["bid", "ask"], errors="ignore")
        #output_file = "output/labeled_trades_with_gex_normal_dropped.csv"
        output_file = file
        df.to_csv(output_file, index=False)

if __name__ == '__main__':
    unittest.main()
