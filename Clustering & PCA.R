#Source: https://www.datanovia.com/en/lessons/k-means-clustering-in-r-algorith-and-practical-examples/
#Data source: https://www.kaggle.com/deepakg/usarrests


data("USArrests")      # Loading the data set
df <- scale(USArrests) # Scaling the data

# View the firt 10 rows of the data
head(df, n = 10)
#The standard R function for k-means clustering is kmeans()
# [stats package], which simplified format is as follow:
#kmeans(x, centers, iter.max = 10, nstart = 1)

# Compute k-means with k = 4
set.seed(123)
km.res <- kmeans(df, 4, nstart = 25)
# Print the results
print(km.res)

aggregate(USArrests, by=list(cluster=km.res$cluster), mean)
dd <- cbind(USArrests, cluster = km.res$cluster)
head(dd)
head(km.res$cluster, 4)


# PCA
library(caret)
library(earth)
library(e1071)
tr_dat <- twoClassSim(100) 
mod <- train(Class ~ ., data = tr_dat, method = "knn",    preProc = c("center", "scale", "pca"))
head(mod$preProcess$trace)
min(which(mod$preProcess$trace > .95))
