### Simplified Neural Network in r
library(ggplot2)
library(dplyr)
library(readxl)
library(Boruta)
library(neuralnet)

df1 <- read_excel("Downloads/hmeq.xls")
colSums(is.na(df1))


df.clean <- na.omit(df1)
glimpse(df.clean)

boruta_output <- Boruta(BAD ~ ., data=df.clean, doTrace=0, maxRuns = 1000)

bs<- getSelectedAttributes(boruta_output, withTentative = TRUE)

plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")
```

unique(df.clean[c("REASON")])


df.clean$REASON <- case_when(
  
  df.clean$REASON == "DebtCon" ~ 1,
  df.clean$REASON == "HomeImp" ~ 0,
)

unique(df.clean[c("JOB")])


df.clean$JOB <- case_when(
  df.clean$JOB == "Self" ~ 5,
  df.clean$JOB == "Sales" ~ 4,
  df.clean$JOB == "ProfExe" ~ 3,
  df.clean$JOB == "Mgr" ~ 2,
  df.clean$JOB == "Office" ~ 1,
  df.clean$JOB == "Other" ~ 0,
)

glimpse(df.clean)


min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

df.n <- as.data.frame(lapply(df.clean, min_max_norm))


sigmoid <- function(x)
{return(1/(1+exp(-x)))}


d.sigmoid <- function(x) 
{return(x*(1-x))}


NNet <- function(X, y, hidden.layers, n, l){
  X <- cbind(rep(1, nrow(X)), X)
  mse <- rep(0, n)
  weight.1 <- matrix(runif(ncol(X)*hidden.layers[1], -1, 1), nrow = ncol(X))
  weight.2 <- matrix(runif((  hidden.layers[1]+1)*hidden.layers[2], -1, 1), nrow = hidden.layers[1]+1)
  weight.3 <- matrix(runif(( hidden.layers[2]+1)*ncol(y), -1, 1), nrow = hidden.layers[2]+1)
  
  for(k in 1:n){
    
    hidden1 <- cbind(matrix(1, nrow = nrow(X)), sigmoid(X %*% weight.1))
    hidden2 <- cbind(matrix(1, nrow = nrow(X)), sigmoid(hidden1 %*% weight.2))
    y_hat <- sigmoid(hidden2 %*% weight.3)
    y_hat_del <- (y-y_hat)*(d.sigmoid(y_hat))
    hidden2_del <- y_hat_del %*% t(weight.3)*d.sigmoid(hidden2)
    hidden1_del <- hidden2_del[,-1] %*% t(weight.2)*d.sigmoid(hidden1)
    
    W3 <- weight.3 + l*t(hidden2) %*% y_hat_del
    W2 <- weight.2 + l*t(hidden1) %*% hidden2_del[,-1]
    W1 <- weight.1 + l*t(X) %*% hidden1_del[,-1]
    mse[k] <- 1/nrow(y)*sum((y-y_hat)^2)
    
    if((k %% (10^4+1)) == 0) cat("mse:", mse[k], "\n")
  }
  
  
  xvals <- seq(1, n, length = 1000)
  
  
  print(qplot(xvals, mse[xvals], geom = "line", main = "MSE", xlab = "Iteration"))
  
  return(print(mean(mse)))
}


X.loan <- as.matrix(df.n[c(9,10,13)])
Y.Bad <- model.matrix(~ BAD- 1, data = df.n)
out.loan <- NNet(X.loan, Y.Bad, hidden.layers= c(6, 6) , n = 5000, l = 0.02)






