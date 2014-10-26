Machine_Learning_Project
========================

Machine Learning Coursera Project
In this work I will analyze the data given by this assignment and by conducting a machine learning process in the training set I will estimate the exercise category in the test set. In order to do so I first describe what I see in the data (control variables and dependent variable as well as missing samples) and I filter it given the high number of missing variables and near zero variance. Moreover, I create a partition to start the machine learning process. In this partition 70 percent of the data will be categorize as the training set while the remaining will be the testing set. Once the data is ready for further process, I apply a Random Forest technique to develop an algorithm that can categorize the data into the different class category of the dependent variable. The performance of this algorithm is sound and accurate in both, the training and the testing data.
Data description:

Once the files have been saved in a cvs format in my computer I proceed to uploaded them:

> library(gdata)

> Test <- read.csv("C:\\Users\\Andres\\Desktop\\Machine lerning\\pml-testing.csv", header=T,sep=";")

> Train <- read.csv("C:\\Users\\Andres\\Desktop\\Machine lerning\\pml-training.csv", header=T,sep=";")

The two data sets given in this exercise are the “Train” set, which is the one that I will use for my analysis and the “Test” set which I will use to validate the performance of my algorithm. Since the data is split with “;” I need to specify it.

By typing the following commands, we get a better insight of how this data looks like:

> dim(Train)

[1] 19622   160

> dim(Test)

[1]  20 160

This shows that the Train data set is built with 160 variables and 19622 observations while the Test data set is built with the same amount of variables and only 20 observations.

Moreover, to see the composition of the dependent variable (class) I type the following command.

> table(Train$classe)

   A    B    C    D    E

5580 3797 3422 3216 3607

The dependent variable is composed of  5 categories corresponding to different types of conducting a weight lifting exercise.  

 Processing the data:

In this section I create a partition of the data into training and testing set and moreover I remove the missing variables and the near zero variance. Recall that the near zero variance variables are those variables with really little variability which will likely not be a good predictors, which motivates removing them.  

> library(caret)

> library(ggplot2)

> library(lattice)

> library(kernlab)

> library(randomForest)

>  inTrain <- createDataPartition(Train$classe,p=.7,list=FALSE)

> training = Train[inTrain,]

> testing = Train[-inTrain,]

> set.seed(143759)

With this partition 70% of my data from the “Train” data set is under the “training” object while the remaining is under the “testing” object.  I have set the seed to be 143759 which is totally random.

I will now proceed to exclude the near zero variance features from the training set. 

> nzvcol <- nearZeroVar(training)

> training <- training[, -nzvcol]

Furthermore, and given the high amount of missing variables (NA) that we can observe in the data by typing “Test” or “Train” it makes sense the get rid of those variables with more that 50 percent of the observations missing:

> cntlength <- sapply(training, function(x) {

>   sum(!(is.na(x) | x == ""))

> })

> nullcol <- names(cntlength[cntlength < 0.5 * length(training$classe)])

> descriptcol <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2",

>     "cvtd_timestamp", "new_window", "num_window")

> excludecols <- c(descriptcol, nullcol)

> training <- training[, !names(training) %in% excludecols]

We now have a clean data, where those variables with more that 50% observations missing are not longer in our sample set and neither those with near zero variance.

Random Forest and Model validation:

Since the variable that we want to predict (class) is composed by 5 different types (A, B, C, D & E) it makes sense to use a Random Forest Process. This process operates by constructing a multitude of decision trees and outputting the class in individual branches. In other words, the uncorrelated trees are composed with specific algorithms with high predictor power for each class. In this specific example, I construct 15 specific trees (random number) or partitions that help determine what observations correspond to each category.

One way to start with Random Forest is to create a modFit function:

>modFit <- train(training$classe ~.,method="rf",prox=TRUE)

>print(modFit)

CART

13737 samples

   52 predictor

    5 classes: 'A', 'B', 'C', 'D', 'E'

No pre-processing

Resampling: Bootstrapped (25 reps)

Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ...

Resampling results across tuning parameters:

  cp      Accuracy  Kappa  Accuracy SD  Kappa SD

  0.0338  0.501     0.347  0.0435       0.0720 

  0.0603  0.401     0.183  0.0589       0.0979 

  0.1148  0.327     0.064  0.0420       0.0628 

With this commands I have create the model composed of 13737 samples, 52 predictor and 5 classes. By setting prox=True, the function produces a little more extra information. Nevertheless to use a rfModel function might be more helpful since the results are going to be easier to interpret it. In the following command I set the rfModel function with 15 different trees.

> rfModel <- randomForest(classe ~ ., data = training, importance = TRUE, ntrees = 15)

The next step is to show how well this algorithm predicts the data in which is built from.

> ptraining <- predict(rfModel, training)

> print(confusionMatrix(ptraining, training$classe))

Which leads to the following results:

Confusion Matrix and Statistics

          Reference

Prediction    A    B    C    D    E

         A 3906    0    0    0    0

         B    0 2658    0    0    0

         C    0    0 2396    0    0

         D    0    0    0 2252    0

         E    0    0    0    0 2525

 

Overall Statistics                                   

               Accuracy : 1         

                 95% CI : (0.9997, 1)

    No Information Rate : 0.2843    

    P-Value [Acc > NIR] : < 2.2e-16 

                                    

                  Kappa : 1         

 Mcnemar's Test P-Value : NA        

 

Statistics by Class:

 

                     Class: A Class: B Class: C Class: D Class: E

Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000

Specificity            1.0000   1.0000   1.0000   1.0000   1.0000

Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000

Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000

Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838

Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838

Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838

Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

The accuracy of 1, the constrained confident interval and the close to cero P-value tells us that the prediction power of this algorithm is really strong and highly significant. Nevertheless, and in order to test for over fitting, we now need to apply it to the “testing” set.

> ptraining <- predict(rfModel, training)

> print(confusionMatrix(ptraining, training$classe))

Confusion Matrix and Statistics

 

          Reference

Prediction    A    B    C    D    E

         A 1673   13    0    0    0

         B    1 1122    3    0    0

         C    0    4 1023   10    2

         D    0    0    0  953    2

         E    0    0    0    1 1078

 

Overall Statistics

                                         

               Accuracy : 0.9939         

                 95% CI : (0.9915, 0.9957)

    No Information Rate : 0.2845         

    P-Value [Acc > NIR] : < 2.2e-16      

                                         

                  Kappa : 0.9923         

 Mcnemar's Test P-Value : NA             

 

Statistics by Class:

 

                     Class: A Class: B Class: C Class: D Class: E

Sensitivity            0.9994   0.9851   0.9971   0.9886   0.9963

Specificity            0.9969   0.9992   0.9967   0.9996   0.9998

Pos Pred Value         0.9923   0.9964   0.9846   0.9979   0.9991

Neg Pred Value         0.9998   0.9964   0.9994   0.9978   0.9992

Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839

Detection Rate         0.2843   0.1907   0.1738   0.1619   0.1832

Detection Prevalence   0.2865   0.1913   0.1766   0.1623   0.1833

Balanced Accuracy      0.9982   0.9921   0.9969   0.9941   0.9980

As expected, the accuracy in this set is slightly lower (0.99) but nevertheless really powerful and accurate (look at P-values and confident interval). 

Once we have validate the machine learning algorithm given by the Random Forest exercise in partitioned sets from the “Train” data, it is time to predict the categories of those observation from the “Test” data set.

> ptest <- predict(rfModel, Test)

> ptest

 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20

 B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B

Levels: A B C D E

The results above tell us to what class belongs each of the 20 different observations.
