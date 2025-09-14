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

    def test_win_rate_by_potential_return_quantiles(self):
        """
        Analyze win rate across different quantiles of potentialReturnAnnal
        """
        df_enriched = self.df_enriched.copy()
        
        # Add labelled_winner column if not exists
        if 'labelled_winner' not in df_enriched.columns:
            df_enriched['labelled_winner'] = (df_enriched['return_pct'] > 0).astype(int)
        
        # Define quantile thresholds
        quantiles = [30, 40, 50, 60, 70, 90, 120, 150, 200]
        
        # Calculate quantile values
        quantile_values = []
        for q in quantiles:
            if q <= 100:
                quantile_val = df_enriched['potentialReturnAnnual'].quantile(q/100)
            else:
                # For values > 100, interpret as absolute thresholds
                quantile_val = q
            quantile_values.append(quantile_val)
        
        # Create segments
        segments = []
        prev_val = df_enriched['potentialReturnAnnual'].min()
        
        for i, (q, val) in enumerate(zip(quantiles, quantile_values)):
            if q <= 100:
                # Quantile-based segment
                mask = (df_enriched['potentialReturnAnnual'] > prev_val) & (df_enriched['potentialReturnAnnual'] <= val)
                segment_name = f"Q{q} ({prev_val:.1f} to {val:.1f})"
            else:
                # Absolute threshold segment
                if i == 0 or quantiles[i-1] <= 100:
                    # First absolute threshold or transition from quantile
                    if i > 0:
                        prev_val = quantile_values[i-1]
                mask = (df_enriched['potentialReturnAnnual'] > prev_val) & (df_enriched['potentialReturnAnnual'] <= val)
                segment_name = f"{prev_val:.1f} to {val:.1f}"
            
            segment_data = df_enriched[mask]
            
            if len(segment_data) > 0:
                win_rate = segment_data['labelled_winner'].mean() * 100
                count = len(segment_data)
                wins = segment_data['labelled_winner'].sum()
                
                segments.append({
                    'Segment': segment_name,
                    'Min': segment_data['potentialReturnAnnual'].min(),
                    'Max': segment_data['potentialReturnAnnual'].max(),
                    'Count': count,
                    'Wins': wins,
                    'Win Rate (%)': win_rate
                })
            
            prev_val = val
        
        # Add final segment for values above the last threshold
        mask = df_enriched['potentialReturnAnnual'] > quantile_values[-1]
        segment_data = df_enriched[mask]
        
        if len(segment_data) > 0:
            win_rate = segment_data['labelled_winner'].mean() * 100
            count = len(segment_data)
            wins = segment_data['labelled_winner'].sum()
            
            segments.append({
                'Segment': f"> {quantile_values[-1]:.1f}",
                'Min': segment_data['potentialReturnAnnual'].min(),
                'Max': segment_data['potentialReturnAnnual'].max(),
                'Count': count,
                'Wins': wins,
                'Win Rate (%)': win_rate
            })
        
        # Create DataFrame and display results
        results_df = pd.DataFrame(segments)
        
        print("\n" + "="*80)
        print("Win Rate Analysis by potentialReturnAnnual Segments")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Summary statistics
        print("\n" + "-"*80)
        print("Summary Statistics:")
        print(f"Total samples: {df_enriched.shape[0]}")
        print(f"Overall win rate: {df_enriched['labelled_winner'].mean()*100:.2f}%")
        print(f"potentialReturnAnnual range: [{df_enriched['potentialReturnAnnual'].min():.2f}, {df_enriched['potentialReturnAnnual'].max():.2f}]")
        
        # Save results to CSV
        results_df.to_csv('test/data/win_rate_by_potential_return_segments.csv', index=False)
        print("\nResults saved to: test/data/win_rate_by_potential_return_segments.csv")
        
        return results_df


if __name__ == '__main__':
    unittest.main()
