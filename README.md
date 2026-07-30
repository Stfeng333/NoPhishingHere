# How to run
- Create a virtual environment and install dependencies from requirements.txt
- Download the email logs database to test (the database must have the fields: sender, subject, body).
- Download the top most trusted domains databases (these will be used to compare domains to help the model catch phishing emails).
- Run your original database through cleaningground.py to clean and parse the email logs, it will also generate a new csv for you to use later (you need to put in the file name manually in the code).
- Pass the new clean database generated from the recent step through fusiontesting.py (you need to put in the file name manually in the code).
- It takes around 10-20 minutes to run the database through the model to get results.
- The output will be the results of categorizing the emails, and the performance metrics. 
- The model shows you 3 methods used (only N-grams/TF-IDF, only BERT, and combination between Tabular features + N-grams/TF-IDF + BERT).

# General steps of our project
- The team used the following datasets from Kaggle: CEAS_08.csv, Nigerian_Fraud.csv, SpamAssasin.csv. Credits to "Naser Abdullah Alam. (2024). Phishing Email Dataset [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/5074342"
- Additional citations: *Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024, May 19). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. ArXiv.org. https://arxiv.org/abs/2405.11619*
- The datasets chosen to have a baseline for trusted domainds and TLDs: https://www.gigasheet.com/sample-data/fortune-500-email-domains, https://github.com/fffaraz/datasets/blob/master/top500Domains.csv.
- The main dataset used to develop the cleaning/parsing script was CEAS_08.csv
- Hence, we developed the cleaning/parsing script to handle email logs that with the format presented on CEAS_08.csv
- Then the new clean dataset generated runs through finalfusion.py, which outputs the result of the model testing the use of only N-grams/TF-IDF, the use of only BERT, and finally the fusion between (Tabular + Ngrams/TF-IDF + BERT)
- There are different percentages in the performance metrics on each of the runs, and distinct results in the email classification results. The differences are minimal, demonstrating that the model is learning and thinking rather than taking strict decisions out of memorization.

