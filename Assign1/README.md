# Group Members:
•	Ali Muhammad (Leader - 64413) 
•	Syed Muhammad Tanzeel (60790)
•	Muhammad Mughees (62749)

# Description:
1.	We have two files train.csv & test.csv. Now we read train.csv file and split it into two parts which is ‘x’ and ‘y’, where ‘y’ has a label column and ‘x’ has remaining 784 columns of pixels.
2.	Then we cross validate (CV) it from train_x and train_y to test_x, test_y because it will split our training data into 20% test and 80% of train.
3.	Now we will applying classification problem from Scikit team (which is Categorial Naïve Bayes) on 20% of test data which were cross validate later.
4.	After applying Categorial Naïve Bayes, we achieved score 0.95 or 95% and predict our labels.
5.	Then we implement our model on test.csv from Kaggle. We used tensor flow documentation because it is easy to understand.
