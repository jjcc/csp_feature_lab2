# Process
* Common steps for training/testing
  * define input, output and other parameters in .env
  * ~~run "a00buld_basic_datasets.py" to collect data from folders~~
  * ~~run "a01merge_gex_macro_features.py" to add mandatary features~~
  * run "a00buld_basic_datasets_with_features.py" to collect data from folders and add mandatary features
  * (in future) run "a03_merge_fundementals_events.py" to add more features
  * (added a new optional workflow) incremental merge new test dataset of latest 2 weeks into traning dataset
    * run "a07merge_dataset_with_features.py
  * run "a09label_data.py" to label the data
    * based on if incremental merge dataset is used, use flag to control with labeling workflow is run
* Training:
  * ~~run "tran_winner_classifier_pct_oof.py" to train the model~~
  * run "b01tran_winner_classifier_pct_oof.py" to train the model
* Testing:
  * run "score_winner_classifer_env.py" to score the data

# Iteration
* With time passing by, the model needs to be retrained with new data to keep update
  * A methodology need to manage and track models/training/verification
  * yaml based configuration with variable controlled current training/testing already implemented
  * It needs to have a name convension
  * It might need an option of  Weekday/daysToExpiry filter

# Notes:
* folder option/put/unprocessed (2025 Apr 25 - Aug 8) as base traning set
* folder option/put/put25_0808-

