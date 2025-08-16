import unittest
import pandas as pd
from service.utils import prep_tail_training_df
from train_tail_with_gex import _prep_df

class TestUtil(unittest.TestCase):

    def test_prep_tail_training_derived(self):
        """
        Compare prep_tail_training_derived and _prep_df outputs
        """
        df = pd.read_csv("output/labeled_trades.csv")
        result1 = prep_tail_training_df(df)
        result2 = _prep_df(df)
        # Check if both functions produce the same result
        pd.testing.assert_frame_equal(result1, result2)

        self.assertIsInstance(result1, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
