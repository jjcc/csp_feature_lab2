import unittest
import joblib
import pandas as pd
from service.preprocess import add_dte_and_normalized_returns
from service.utils import prep_tail_training_df, fill_features_with_training_medians
from service.utils import ALL_FEATS
from train_tail_with_gex import _fill_features

class TestDataAna(unittest.TestCase):

    def setUp(self):
        self.df_enriched = pd.read_csv("output/labeled_trades_enriched.csv")

    
    def test_compare_potentialann_realmon(self):
        """
        compare the columns of basic, gex, and macro feature datasets
        """
        df_enriched = self.df_enriched
        # filter the potentialReturnAnnual < 400
        df_enriched = df_enriched[df_enriched['potentialReturnAnnual'] < 100]
        # lable the return_pct > 0 as 1, else 0
        df_enriched['labelled_winner'] = (df_enriched['return_pct'] > 0).astype(int)



        # add the monthly return column
        df_enriched = add_dte_and_normalized_returns(df_enriched)
        c1 = df_enriched['potentialReturnAnnual']
        c2 = df_enriched['return_mon'] * 12

        # plot the two columns against each other
        import matplotlib.pyplot as plt
        # smaller dot, bitter chart image
        plt.figure(figsize=(8, 6))
        plt.scatter(c1, c2, alpha=0.5, s=5)
        plt.xlabel('potentialReturnAnnual')
        plt.ylabel('return_mon * 12')
        plt.title('Comparison of potentialReturnAnual and return_mon * 12')
        plt.grid(True)
        #plt.plot([c1.min(), c1.max()], [c1.min(), c1.max()], 'r--')  # y=x line
        plt.savefig('test/data/potentialann_vs_realmon_100max_b.png')



        #gex extra = {'gex_pos', 'gex_missing', 'gex_distance_to_flip', 'gex_total_abs', 'symbol_norm', 'gex_sign_at_ul', 'gex_flip_strike', 'gex_file', 'trade_date', 'gex_neg', 'gex_center_abs_strike', 'gex_total', 'gex_gamma_at_ul'}
        #macro extra = {'VIX', 'ret_5d_norm', 'prev_close_minus_strike', 'ret_2d', 'prev_close_minus_strike_pct', 'log1p_DTE', 'prev_close', 'ret_5d', 'ret_2d_norm'}


if __name__ == '__main__':
    unittest.main()
