# Process
* Common steps for training/testing
  * define input, output and other parameters in .env
  * run "a00buld_basic_datasets.py" to collect data from folders
  * run "a01merge_gex_macro_features.py" to add mandatary features
  * (in future) run "a03_merge_fundementals_events.py" to add more features
  * run "a09label_data.py" to label the data
* Training:
  * run "tran_winner_classifier_pct_oof.py" to train the model
* Testing:
  * run "score_winner_classifer_env.py" to score the data