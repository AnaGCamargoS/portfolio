#install packages, load libraries
install.packages("dplyr")
install.packages("class")
install.packages("e1071")
install.packages("grpreg")
library(dplyr) #one hot encoding
library(class) #k-NN
library(e1071) #NaiveBayes
library(grpreg) #GLPL

# import dataframe 
lastname_df <- read.csv("C:\\Users\\avc6470\\OneDrive - The Pennsylvania 
                        State University\\PennState\\Research Paper\\
                        class_last_names.csv")
summary(lastname_df)

# checking missing values and removing
summary(is.na(lastname_df))
lastname_df <- na.omit(lastname_df)  #removing NAs
lastname_df$last_name <- toupper(lastname_df$last_name)  #uppercase
summary(is.na(lastname_df))

# indicator variables code - One hot encoding
namesY <- lastname_df
noRows <- dim(namesY)[1]
nHispanic <- sum(namesY[,2]=="hispanic")
nNonHispanic <- noRows-nHispanic
## Select a subset as  data
sizeTrainTest <- round(nHispanic/2)*2  #total sample size Hisp + nonHisp
maxLength <- max(nchar(namesY$last_name))
indices <- seq(1,noRows)
set.seed(3)
subsetH <- sample(indices[1:12497], size=round(sizeTrainTest/2))
subsetNH <-sample(indices[12498:97093], size=round(sizeTrainTest/2))
p <- nHispanic/noRows #proportion of Hisp in the sample
p
# Sample H and non H names randomly and put them together
S1 <- namesY[subsetH,]
S2 <- namesY[subsetNH,]
S <- rbind(S1,S2)
# Reshuffle final dataframe (S). This contains train+test data
set.seed(3)
S <- S[sample(sizeTrainTest),]
Y <- vector(length=sizeTrainTest)
for(i in 1:sizeTrainTest){
  if (S[i,2]=="hispanic"){
    Y[i] <- 0;
  }else{Y[i]<- 1}
} ##coding Hispanic="0", nonHisp="1"

# create a 1-of-K vector for each character in name and concatenate all of them
X <- matrix(nrow=sizeTrainTest,ncol=33*maxLength,0)
library(dplyr)
for (i in 1:sizeTrainTest){
  #Define a 33xmaxLength matrix for each name, 
  ##we will convert it to a row of matrix X
  xrow <- matrix(nrow=33,ncol=maxLength,0)
  len <- nchar(S[i,1])
  for (j in 1:min(len,maxLength)){
    charToCode <- substr(toupper(S[i,1]),j,j)
    index <- case_match(
      charToCode,
      "A" ~1,
      "B" ~2,
      "C" ~3,
      "D" ~4,
      "E" ~5,
      "F" ~6,
      "G" ~7,
      "H" ~8,
      "I" ~9,
      "J" ~10,
      "K" ~11,
      "L" ~12,
      "M" ~13,
      "N" ~14,
      "Ñ" ~15,
      "O" ~16,
      "P" ~17,
      "Q" ~18,
      "R" ~19,
      "S" ~20,
      "T" ~21,
      "U" ~22,
      "V" ~23,
      "W" ~24,
      "Y" ~25,
      "X" ~26,
      "Z" ~27,
      "Á" ~28,
      "É" ~29,
      "Í" ~30,
      "Ó" ~31,
      "Ú" ~32,
      " " ~33
    )
    xrow[index,j] <- 1
  } #endfor j
  xrow <- c(xrow)
  X[i,]<- t(xrow)
} #endfor i ---> ~3min running time

# split X and Y into train/test
set.seed(3)
TrainRows <- sample.int(sizeTrainTest,sizeTrainTest*2/3,replace = FALSE)
TestRows <- setdiff(seq(1:sizeTrainTest),TrainRows)
Xtrain <- X[TrainRows,]
Ytrain <- Y[TrainRows]
Xtest <- X[TestRows,]
Ytest <- Y[TestRows]

# k-NN for a range of "k" values --> split if necessary (running time)
library(class)
n = 300
knn_error2 = rep(NA,n)
tpr2 = rep(NA,n)
fpr2 = rep(NA,n)
# do 1-100,101-200,201-300
for (m in 1:n){
  set.seed(2)
  knn_pred2 <- knn(Xtrain, Xtest, Ytrain , k=m)
  knn_error2[m] <- mean(knn_pred2 != Ytest)
  t2 <- table(knn_pred2, Ytest)
  tpr2[m] <- t2[1]/(t2[1]+t2[2])
  fpr2[m] <- t2[3]/(t2[3]+t2[4])
}
# tabulating misclassification error, tpr, fpr for all values of k
knn_error_df2 <- data.frame(k = 1:n2 , test_error = knn_error2, 
                            TPR = tpr2, FPR = fpr2)
summary(knn_error_df2)  ##there are NA's, error in model: too many ties in knn
knn_error_df2 <- na.omit(knn_error_df2)  ##removing NA's
summary(knn_error_df2) 
# calculate euclidean distance from point (0,1) to each point
dist_knn <- sqrt(((knn_error_df2$TPR-1)^2)+((knn_error_df2$FPR)^2))
knn_error_df2 <- data.frame(knn_error_df2, dist_knn)
best_k <- which.min(knn_error_df2$dist_knn)  #minimum dist_knn
# ROC curve - using different values of k
plot(knn_error_df2$FPR, knn_error_df2$TPR, xlab="False Positive Rate (FPR)" , 
     ylab="True Positive Rate (TPR)", type="b", 
     main="ROC curve for k-NN (k=1-267)")
points(knn_error_df2$FPR[best_k], knn_error_df2$TPR[best_k], 
       col="red", pch=19)
# adjusting y-axis to 1, and same scale for x&y
plot(knn_error_df2$FPR, knn_error_df2$TPR, xlab="False Positive Rate (FPR)" , 
     ylab="True Positive Rate (TPR)", type="b", 
     main="ROC curve for k-NN (k=1-267)", 
     xlim=c(0.0,0.6), ylim=c(0.5,1.0), asp = 1)
points(knn_error_df2$FPR[best_k], knn_error_df2$TPR[best_k], 
       col="red", pch=19)
points(0,1,col="blue",pch=4)

# predicting with best value of k (~35sec running time)
best_k   # best=7
knn_pred2 <- knn(Xtrain, Xtest, Ytrain, k=best_k)
## confusion matrix using k=best_k=7--> TPR=TP/P FPR=FP/N
t2 <- table(knn_pred2, Ytest)
t2
mean(Ytest == knn_pred2) #Accuracy
t2[1]/(t2[1]+t2[2])      #TPR
t2[3]/(t2[3]+t2[4])      #FPR

# NAIVE BAYES
library(e1071)
XYmerge <- as.data.frame(cbind(Y,X))
nb_fit <- naiveBayes(Y~., data= XYmerge , subset = TrainRows)
nb_pred <- predict(nb_fit , XYmerge[TestRows,])
t3 <- table(nb_pred, Ytest)
t3
mean(Ytest == nb_pred) #Accuracy
t3[1]/(t3[1]+t3[2])    #TPR
t3[3]/(t3[3]+t3[4])    #FPR

# LOGISTIC REGRESSION
lr_fit <- glm(Y~. , family=binomial, data=XYmerge[TrainRows,])
lr_pred <- round(predict(lr_fit , XYmerge[TestRows,], type="response"))   #el round es porque la prediccion de 0 estaba resultando en exp-16
t4 <- table(lr_pred, Ytest)
t4
mean(Ytest == lr_pred) #Accuracy
t4[1]/(t4[1]+t4[2])    #TPR
t4[3]/(t4[3]+t4[4])    #FPR

# GLPL model  ~6min in total
library(grpreg)
# Define groups--groups of indicator variables defining each character in the 
               # sequence, c1 to cmaxLen=17
group <- c(rep("c1",33),rep("c2",33),rep("c3",33),rep("c4",33),
           rep("c5",33),rep("c6",33),rep("c7",33),rep("c8",33),
           rep("c9",33),rep("c10",33),rep("c11",33),rep("c12",33),
           rep("c13",33),rep("c14",33),rep("c15",33),rep("c16",33),
           rep("c17",33))
glpl_fit <- grpreg(Xtrain,Ytrain,group,penalty="grLasso",family="binomial")
plot(glpl_fit)
# Do a 10-fold cross-validation fit to get best lambda
cvglpl_fit <- cv.grpreg(Xtrain, Ytrain, group, penalty="grLasso",
                        family="binomial")
cvglpl_fit$lambda.min #best penalization parameter
plot(cvglpl_fit)
# Now predict at test data and get confusion matrix
glpl_pred <-  predict(cvglpl_fit, Xtest, type="class")
t5 <- table(glpl_pred, Ytest)
t5
mean(Ytest == glpl_pred) #Accuracy
t5[1]/(t5[1]+t5[2])      #TPR
t5[3]/(t5[3]+t5[4])      #FPR

# Alternate method to get lambda: varying lambda from 0.0001 in steps of 0.0001
## approx same result as in CV (0.001 ~ 0.00098) so we keep CV model
gmax = 30   ## gmax = ((t_lam-f_lam)/s_lam) + 1
f_lam = 0.0001
s_lam = 0.0001
t_lam = ((gmax-1)*s_lam)+f_lam   ## valor fijo

glpl2_error = rep(NA,gmax)
glpl2_tpr = rep(NA,gmax)
glpl2_fpr = rep(NA,gmax)
lam = seq(from = f_lam, to = t_lam , by = s_lam)   #initial lambda = 0.01

for (g in 1:gmax){
  glpl2_fit <- grpreg(Xtrain, Ytrain, group, penalty="grLasso", family="binomial",lambda=lam[g])  #model fitting
  glpl2_pred <- predict(glpl2_fit, Xtest, type="class")   #model prediction
  t6 <- table(glpl2_pred, Ytest)    #confusion matrix
  glpl2_error[g]<- mean(glpl2_pred != Ytest)
  glpl2_tpr[g] <- t6[1]/(t6[1]+t6[2])
  glpl2_fpr[g] <- t6[3]/(t6[3]+t6[4])
}

# tabulating misclassification error, tpr, fpr for all values of k
glpl2_error_df <- data.frame(lambda=lam , test_error=glpl2_error, TPR=glpl2_tpr, FPR=glpl2_fpr)
# ROC curve - using different values of k
plot(glpl2_error_df$FPR, glpl2_error_df$TPR, xlab="False Positive Rate" , ylab="True Positive Rate", type="b")
# calculate euclidean distance from point (0,1) to each point
dist_glpl <- sqrt(((glpl2_error_df$TPR-1)^2)+((glpl2_error_df$FPR)^2))
glpl2_error_df <- data.frame(glpl2_error_df, dist_glpl)

# selecting lambda with min distance to (0,1)
best_lam <- lam[which.min(glpl2_error_df$dist_glpl)]
best_lam
# Now predict at test data with best_lam and get confusion matrix
glpl2_fit <- grpreg(Xtrain, Ytrain, group, penalty="grLasso", family="binomial",lambda=best_lam)  #model fitting
glpl2_pred <-  predict(glpl2_fit, Xtest, type="class")
t6 <- table(glpl2_pred, Ytest)
t6
mean(Ytest == glpl2_pred) #Accuracy
t6[1]/(t6[1]+t6[2])       #TPR
t6[3]/(t6[3]+t6[4])       #FPR


# Now classifying in the remaining part of the dataset

used_rows <- c(subsetH, subsetNH)
namesY_new <- namesY[-used_rows,] #new df excluding used last names
size <- nrow(namesY_new)
nHispanic_new <- sum(namesY_new[,2]=="hispanic")
nNonHispanic_new <- size-nHispanic_new
p_new <- nHispanic_new/size  #proportion of Hispanic in new dataset
p_new

S_new <- namesY_new[sample(size),]  #reshuffle df
Y_new <- vector(length=size)
for(i in 1:size){
  if (S_new[i,2]=="hispanic"){
    Y_new[i] <- 0;
  }else{Y_new[i]<- 1}
} ##coding Hispanic="0", nonHisp="1"

#One hot encoding ---> ~11min running time for 85k
X_new <- matrix(nrow=size,ncol=33*maxLength,0)
library(dplyr)
for (i in 1:size){
  #Define a 33xmaxLength matrix for each name, 
  ##we will convert it to a row of matrix X
  xrow <- matrix(nrow=33,ncol=maxLength,0)
  len <- nchar(S_new[i,1])
  for (j in 1:min(len,maxLength)){
    charToCode <- substr(toupper(S_new[i,1]),j,j)
    index <- case_match(
      charToCode,
      "A" ~1,
      "B" ~2,
      "C" ~3,
      "D" ~4,
      "E" ~5,
      "F" ~6,
      "G" ~7,
      "H" ~8,
      "I" ~9,
      "J" ~10,
      "K" ~11,
      "L" ~12,
      "M" ~13,
      "N" ~14,
      "Ñ" ~15,
      "O" ~16,
      "P" ~17,
      "Q" ~18,
      "R" ~19,
      "S" ~20,
      "T" ~21,
      "U" ~22,
      "V" ~23,
      "W" ~24,
      "Y" ~25,
      "X" ~26,
      "Z" ~27,
      "Á" ~28,
      "É" ~29,
      "Í" ~30,
      "Ó" ~31,
      "Ú" ~32,
      " " ~33
    )
    xrow[index,j] <- 1
  } #endfor j
  xrow <- c(xrow)
  X_new[i,]<- t(xrow)
} #endfor i

# Predicting using trained models and new data

#Naive Bayes (nb_fit) ~7min
library(e1071)
XYmerge_new <- as.data.frame(cbind(Y_new,X_new))
nb_pred_new <- predict(nb_fit , XYmerge_new)
t8 <- table(nb_pred_new, Y_new)
t8
mean(Y_new == nb_pred_new) #Accuracy
t8[1]/(t8[1]+t8[2])        #TPR
t8[3]/(t8[3]+t8[4])        #FPR

#Logistic Reg (lr_fit) ~15sec
lr_pred_new <- round(predict(lr_fit, XYmerge_new, type="response"))
t9 <- table(lr_pred_new, Y_new)
t9
mean(Ytest == lr_pred) #Accuracy
t9[1]/(t9[1]+t9[2])    #TPR
t9[3]/(t9[3]+t9[4])    #FPR

#GLPL CV (cvglpl_fit) ~2sec
library(grpreg)
glpl_pred_new <- predict(cvglpl_fit, X_new, type="class")
t10 <- table(glpl_pred_new, Y_new)
t10
mean(Y_new == glpl_pred_new) #Accuracy
t10[1]/(t10[1]+t10[2])       #TPR
t10[3]/(t10[3]+t10[4])       #FPR

#kNN (k=7) we need train and test sets ~45min
library(class)
set.seed(3)
train <- sample(1:size, round(size*2/3))
test <- -train
knn_pred_new <- knn(X_new[train,], X_new[test,], Y_new[train], k=7)
t11 <- table(knn_pred_new, Y_new[test])
t11
mean(Y_new[test] == knn_pred_new) #Accuracy
t11[1]/(t11[1]+t11[2])            #TPR
t11[3]/(t11[3]+t11[4])            #FPR


#Creating a function to classify (using GLPL) a specific last name 
# --> type lastname and run 408-409
clasif <- function(lastname){
  library(dplyr)
  X_lastname <- matrix(nrow=1,ncol=33*maxLength,0)
  xrow <- matrix(nrow=33,ncol=maxLength,0)
  len <- nchar(lastname)
  for (j in 1:min(len,maxLength)){
    charToCode <- substr(toupper(lastname),j,j)
    index <- case_match(
      charToCode,
      "A" ~1,
      "B" ~2,
      "C" ~3,
      "D" ~4,
      "E" ~5,
      "F" ~6,
      "G" ~7,
      "H" ~8,
      "I" ~9,
      "J" ~10,
      "K" ~11,
      "L" ~12,
      "M" ~13,
      "N" ~14,
      "Ñ" ~15,
      "O" ~16,
      "P" ~17,
      "Q" ~18,
      "R" ~19,
      "S" ~20,
      "T" ~21,
      "U" ~22,
      "V" ~23,
      "W" ~24,
      "Y" ~25,
      "X" ~26,
      "Z" ~27,
      "Á" ~28,
      "É" ~29,
      "Í" ~30,
      "Ó" ~31,
      "Ú" ~32,
      " " ~33
    )
    xrow[index,j] <- 1
  }
  xrow <- c(xrow)
  X_lastname[1,]<- t(xrow)
  
  if (predict(cvglpl_fit, X_lastname, type="class")==0){
    print("This last name is Hispanic")
  } else {
    print("This last name is non-Hispanic")
  }
}

#Type last name inside ""
lastname <- "sandoval"
clasif(lastname)
