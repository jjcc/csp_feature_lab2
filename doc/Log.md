# Some notes
* There's no ts1, ts1 is original traning dataset of 38k recoreds
* 2025-09-28: Merge ts2 (Aug 8 to Sep 5) to original traning
  * Create 2 datasets: merge, and merge then remove same amount. First dataset has 44k+, second has the same 38k
  * The model trained with the first dataset improved on ts3, the model is in "v7_oof_ne_ts_w_lgbm_tr_ts", score in "v7_lgbm_tr_ts1_score_t3"
  * The model trained with the secode dataset improved on ts3, the model is in "v7_oof_ne_ts_w_lgbm_tr_ts_minus", score in "v7_lgbm_tr_ts1_minus_score_t3"
  * Both models are better in test agains ts3
  * The first model is a bit better than the 2nd one, with AUC-ROC: 0.635681 AUC-PRC: 0.951263 vs AUC-ROC: 0.628571 AUC-PRC: 0.949811
* 2025-09-30: Compared the score winner result with the on fly "task_score_tail_winner.py".
  * There was a mistake that use different model to compare.  
  * Initially the GEX was considered to be the cause because task scorer use real time GEX as oppose to "11:00" GEX
  * After fix the GEX, later it was found different model was used. "v7_oof_ne_ts_w_lgbm_tr_ts_minus" caused some wider spread
  * There is still small difference with the same model, same dataset. Reason unknown but difference is small enough
  * The GEX investigation gives some hint: GEX doesn't have very big impact.
  * A new GEX scraping strategy that check if the GEX is collected within 60 min is developed. To reduce the freq to pull GEX and save time
* 2025-10-05
* Due to price management improvement, the "vol20" in sevice/data_prepare.py might be different than preious one.
* A retrain/rescore is done
* 2025-10-12: The deteriorated performance of model was investigated but due to vocation in Kitchner, the fix is not done  
  * **There are 2 causes: cutoff date needs to be consistant, 3-day backfill needes to be removed**
* 2025-10-13: Trying to fix cause 1:
  * Calculated the cutoff date as min(test.expirationDate)
  * Added it into the config.yml
* 2025-10-19: Added anothe project(Not relevant): scrap unusual option from stocknear.com
* 2025-10-20: The call/put option catpure script was broken due to change in the Web. Fixed that project
* 2025-10-31: With collected prod/output data, I'd like to verify the result with the Monday/Weekly result of prediction
* 2025-11-01: Already built utility in test_verification.py with: 1.move Monday data, 2.Fill expiry close price 3. Fill the PnL and win verdict on result
* 2025-11-11: Fixed 3d back fill and cutoff date with advice
* 2025-11-15: Added 1-14 filter and  merge multipe dataset script 

* 2025-12-17: Start fixing low ROC score. Shuffle is found to be a issue caused false high score in training. 
* 2025-12-19: Refactor to make sure training use the same feature extraction function as score and production
* 2025-12-20: With comparison of turn on and off shuffling, it turned out 0.79 of ROC disappear with no shuffling of 0.5+
* 2025-12-21: Continue fixing the duplicates:
    * Focus on Monday/Tuesday and DTE of 4 and 3 for the first step. Total avaialbe data from 38k+ down to 5.4k+
    * keep 1 contract per day , which was fulfilled by using "Force Picking 1 file close to 11:00". Now using all files but among all record choose record closest to 11:00
    * Increased the data by this measure to 5.4k to 7.9k (now it seems to be 10k)
  2025-12-22: With investigation of losing contract, a couple of types should be excluded:
    * Pharma company that could failed on drug test
    * Earning close to expiry date
    * A nasdaq API script is developed to retrieve earning date. Polygon is better. Yfinance is not good
  
