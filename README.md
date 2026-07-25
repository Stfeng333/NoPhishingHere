# Steven side
- The script clean_ceas08 is only to clean the dataset, it does not create another dataset.
- The cleaning that I did was: strip whitespaces, handle missing values (missing values are replaced with NaN), and cleaned the noise in the body text.
- Before I cleaned the spam of symbols in the body text, I did some feature engineering. The feature column "symbol_count" to count total special characters. The feature column "symbol_ratio" that performs "symbol_count / total_char_count" to give us the symbol density relative to the email length.
- Finally, I converted the Subject and Body texts to lowercase.
- You can visualize how the dataset looks like after running my script
- On Saturday I will work on feature engineering with you (I will most likely generate n-grams)


# Henry side