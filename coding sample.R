#### Project application of dimensionality reduction and cluster analysis
#### Marcos Torres Vivanco

library(readr) ## read data
library(scatterplot3d) ## 3D graph
library(Rtsne) ## t-sne algorithm

## We load the MNIST-like dataset of 28x28 labeled fashion images
fashion_mnist_test <- read_csv("fashion-mnist_test.csv")
lab <- as.numeric(t(fashion_mnist_test[,1])) ## label of data
data <- fashion_mnist_test[,-1] ## images

## example of one image
v <- rev(as.numeric(data[10,]))
v1 <- t(matrix(v,ncol=28,byrow=T))
par(mar=c(0,0,0,0))
image(v1,col=grey(seq(0,1,length=256)))

## filtering the shoes from the data
shoes <- subset(fashion_mnist_test,label==7)
sh <- shoes[,-1]

## we use T-SNE to find a 2D representation of the shoes images 
set.seed(20)
m7 <- as.matrix(sh)
mitsne <- Rtsne(m7, pca=TRUE, perplexity=40,theta=0.0,check_duplicates = FALSE)
plot(mitsne$Y,main="T-sne fashion MNIST shoes")

## we find clusters in the cloud of points
m <- as.data.frame(mitsne$Y)
clus <- kmeans(m,3)
plot(m,col=clus$cluster)
nube <- data.frame(clus$cluster,m)
shclust <- data.frame(clus$cluster,sh)

## We obtain samples from each cluster
m1 <- which(shclust$clus.cluster==1)
s1 <- sample(m1,6)

m2 <- which(shclust$clus.cluster==2)
s2 <- sample(m2,6)

m3 <- which(shclust$clus.cluster==3)
s3 <- sample(m3,6)

## the data we choose in each cluster
plot(m,col=clus$cluster)
points(m[s1,],pch=20,lwd=20,col=1)
points(m[s2,],pch=20,lwd=20,col=2)
points(m[s3,],pch=20,lwd=20,col=3)

## images of the samples from each cluster
par(mfrow=c(2,3))
for (i in s1) {
  v <- rev(as.numeric(sh[i,]))
  v1 <- matrix(v,ncol=28)
  par(mar=c(0,0,0,0))
  image(v1,col=grey(seq(0,1,length=256)))
}

par(mfrow=c(2,3))
for (i in s2) {
  v <- rev(as.numeric(sh[i,]))
  v1 <- matrix(v,ncol=28)
  par(mar=c(0,0,0,0))
  image(v1,col=grey(seq(0,1,length=256)))
}

par(mfrow=c(2,3))
for (i in s3) {
  v <- rev(as.numeric(sh[i,]))
  v1 <- matrix(v,ncol=28)
  par(mar=c(0,0,0,0))
  image(v1,col=grey(seq(0,1,length=256)))
}

## we can notice that TSNE identify particularities from each cluster
## for example the second cluster has boot like shoes


































