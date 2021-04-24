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
6.  We did filter/convolution 2D on (28,28,1) -> grey scale because 784= 28x28 we break it and applied (3,3) and (2,2) filter and simultaneously we used dropout which is used as      regularize.
  
7.  Run our model to check on each epochs that the desired accuracy achieved which is 0.990.
8.  Now run and predict our model on test.csv and make a submission file which has Image Id and label and submit the file .csv on Kaggle.
9.  Finally, we got the score of 0.99082 on Kaggle.


# What we learned from this Assignment?

1. We learned that, how to train and test our data on any classification problem and simply predict our digit/labels. 
2. After applying convolution and filter on test and training data, we see that it will get better accuracy and score.
3. Let’s suppose we got score of 0.95% after applying Categorial Naïve Baser and then fit our model on test data and then predict labels so we can achieve 0.990 accuracy.
4. We also learned how to implement hot-encoding on each number. For example: we have a digit 9 so:
                                    000000001
   on bit of 9th index remaining bits are off.  Through this one hot encoding we can evaluate of original training data and no. of labels (predicted Data), this one hot encoding we use/follow tensor flow documention.
5. We are applying Categorical naive Bayes, it is the classifier is suitable for classification with discrete features that are categorically distributed.
6. This model is expected to be called many times consecutively on different chunks of a dataset.
7. This model is useful when the whole dataset is too big to fit in memory at once.

