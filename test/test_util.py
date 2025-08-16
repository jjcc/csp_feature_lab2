import unittest
import joblib
import pandas as pd
from service.utils import prep_tail_training_df, fill_features_with_training_medians
from service.utils import ALL_FEATS
from train_tail_with_gex import _prep_df, _fill_features

class TestUtil(unittest.TestCase):

    def test_prep_tail_training_derived(self):
        """
        Compare prep_tail_training_derived and _prep_df outputs
        """
        df = pd.read_csv("output/labeled_trades_with_gex.csv")
        result1 = prep_tail_training_df(df)
        result2 = _prep_df(df)
        # Check if both functions produce the same result
        pd.testing.assert_frame_equal(result1, result2)

        

        #
        MODEL_IN = "models/tail_model_gex_v2_cut05.pkl"
        pack = joblib.load(MODEL_IN)
        med  = pack["medians"]
        score_x, medians_x = fill_features_with_training_medians(result1, ALL_FEATS)
        score_t, medians_t = _fill_features(result2, ALL_FEATS)

        pd.testing.assert_frame_equal(score_x, score_t)
        assert(medians_x == medians_t)

        self.assertIsInstance(result1, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
