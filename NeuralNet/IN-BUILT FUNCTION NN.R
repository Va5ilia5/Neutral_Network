
# IN-BUILT NEURALNET() in r

library(dplyr)
library(readxl)
library(Boruta)
library(neuralnet)


df1 <- read_excel("Downloads/hmeq.xls")
colSums(is.na(df1))

df.clean <- na.omit(df1)
glimpse(df.clean)

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

boruta_output <- Boruta(BAD ~ ., data=df.clean, doTrace=0, maxRuns = 1000)

bs<- getSelectedAttributes(boruta_output, withTentative = TRUE)

plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")

min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

df.n <- as.data.frame(lapply(df.clean, min_max_norm))


head(df.n)

set.seed(12332) 

bar <- floor(0.66 * nrow(df.n)) 
partition <- sample(seq_len(nrow(df.n)), size = bar) 
train <- df.clean[partition, ] 
test <- df.clean[-partition, ]


nn.loan <- neuralnet(BAD ~ .,
                     data = df.clean,
                     hidden = c(6,4),
                     linear.output = F,
                     lifesign = 'full',
                     rep=1)


plot(nn.loan,
     col.hidden = 'darkgreen',
     col.hidden.synapse = 'darkgreen',
     show.weights = F,
     information = F,
     fill = 'lightblue', rep = "best")

output <- compute(nn.loan, train)
p1 <- output$net.result
pred1 <- ifelse(p1>0.5, 1, 0)
tab1 <- table(Prediction =  pred1, Actuals = train$BAD)


paste0("Misclassification Error of training data: ", round(100 - sum(diag(tab1))/sum(tab1)*100,2))

paste0("Accuracy of training data: ", round(sum(diag(tab1))/sum(tab1) * 100,2))

