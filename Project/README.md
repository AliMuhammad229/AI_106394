# Group Members:

StdID | Name
------------ | -------------
**64413** | **Ali Muhammad (Lead)** 
62749 | Muhammad Mughees
60790 | Syed Muhammad Tanzeel


# Description: How we Achieved each Task?


In this project, we implement four classification techniques


 1.  Multinomial Naïve Bayes
 2.  Linear Regression 
 3.  SUM 
 4.  KNN 

In each technique, we are using 3 convolution 5x5,7x7,9x9 on two different filters.
   1. We have 784 columns of pixels, it breaks into 28x28, then we create a new array of new dimensions/size (2D Array) and apply 5x5 convolution.
   2. After the implementation of 5x5 convolution, new array will became in the shape of (0, 576) because this 5x5 convolution will implemented on 42000                 rows of train.csv





# CV Score of Each Six Techniques: #

## 1. Multinomial Naive Bayer: ##

1. Through this, we can achieve a score on Multinomial Naive Buyer 5x5 convolution with same size filter (1, 1) is 0.76.
2. On 5x5 convolution, with different size filter (average filter) will be achieved 0.77.
3. Now apply 7x7 convolution (49 pixels) with same filter on Multinomial Naïve Bayer, it will get 0.71 score and it will not maintain the score as compare to 5x5      convolution.
4. On 7x7 convolution, with average filter (diff filter) achieved 0.72.
5. After 7x7 convolution, new array will become 0.484.
6. Now applying 9x9 convolution (81 pixels), with same filter on this model it will get 0.67 because when on the convolution, it is increasing and it will not        maintain accuracy.
7. On 9x9 convolution, with average filter (diff filter) we achieve 0.6835.
8. After the application of 9x9, convolution new array will become in the array of (0,400).

