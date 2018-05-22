#Set working directory
setwd("~/Documents/GitHub/plastNN/")

#Load libraries
library(data.table)
library(Biostrings)
library(ggplot2)
library(rDNAse)
library(caret)

#List 20 amino acids
aa = c("A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L","K", "M", "F", "P", "S", "T", "W", "Y", "V")

####################################################################

##Define functions

#Define function to count frequency of amino acids
get_aa_freq = function(x){
  xSplit = strsplit(x, split = "")[[1]]
  aa_counts = table(factor(xSplit, levels = aa))
  aa_freq = aa_counts/sum(aa_counts)
  return(aa_freq)
}

#Define function to count amino acid frequencies in a specific interval
getFreqsinWindow = function(x, start, stop){
  window_seqs = lapply(x[,seq], function(x)substr(x, start, stop))
  x1 = as.data.table(do.call(rbind, lapply(window_seqs, get_aa_freq)))
  return(x1)
}

#Define function to evaluate cross-validation metrics
evaluate = function(model){
  cm = as.data.table(model$resampledCM)
  #For glmnet
  if("alpha" %in% colnames(cm)){
    cm = cm[alpha == model$finalModel$tuneValue$alpha & lambda ==model$finalModel$tuneValue$lambda, .(cell1, cell2, cell3, cell4)]
  }
  #For glm
  cm = cm[,.(cell1, cell2, cell3, cell4)]
  setnames(cm, c("TN", "FP", "FN", "TP"))
  cm[, accuracy:=(TP+TN)/(TP+FP+TN+FN), by=1:nrow(cm)]
  cm[, ppv := TP/(TP+FP), by=1:nrow(cm)]
  cm[, recall := TP/(TP+FN), by=1:nrow(cm)]
  cm[, mcc := ((TP*TN) - (FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)), by=1:nrow(cm)]
  cm[, npv := TN/(TN+FN), by=1:nrow(cm)]
  cm[, specificity := TN/(TN+FP), by=1:nrow(cm)]
  cm_mean = cm[, .(mean(accuracy), mean(ppv), mean(recall), mean(mcc), mean(npv), mean(specificity))]
  cm_sd = cm[, .(sd(accuracy), sd(ppv), sd(recall), sd(mcc), sd(npv), sd(specificity))]
  cat("Accuracy: ", round(cm_mean$V1,2),  " | PPV: ", round(cm_mean$V2,2), 
      " | Recall: ", round(cm_mean$V3,2), " | MCC: ", round(cm_mean$V4,2),
      " | NPV: ", round(cm_mean$V5,2), " | Specificity: ", round(cm_mean$V6,2), "\n")
  cat("Accuracy (SD): ", round(cm_sd$V1,2),  " | PPV: ", round(cm_sd$V2,2), 
      " | Recall: ", round(cm_sd$V3,2), " | MCC: ", round(cm_sd$V4,2), 
      " | NPV: ", round(cm_sd$V5,2), " | Specificity: ", round(cm_sd$V6,2), "\n")
}
###########################################################
##Read datasets

#Read sequence data
pos = fread("data/train_data/positive.txt")
neg = fread("data/train_data/negative.txt")

#Get transcriptome
pos_rna_intervals = fread("data/train_data/pos_rna.txt")
neg_rna_intervals = fread("data/train_data/neg_rna.txt")

#Get tp start
pos_tp = fread("data/train_data/pos_tp.txt")
neg_tp = fread("data/train_data/neg_tp.txt")

###########################################################
##Format sequence data

#Add type of example
pos[, type:="pos"]
neg[, type:="neg"]

#Add transcriptome
pos = merge(pos, pos_rna_intervals, by = "id")
neg = merge(neg, neg_rna_intervals, by = "id")

#Add TP
pos = merge(pos, pos_tp, by = "id")
neg = merge(neg, neg_tp, by = "id")

#Combine positive and negative examples
train = rbind(pos, neg)

#Remove the SP
for(i in 1:nrow(train)){train$seq_cut[i] = substr(train$seq[i], train$first_aa_of_tp[i], nchar(train$seq[i]))}
for(i in 1:nrow(test)){test$seq_cut[i] = substr(test$seq[i], test$first_aa_of_tp[i], nchar(test$seq[i]))}

#Select columns for test and train data tables
train = train[, .(id, seq_cut, type, hr5, hr10, hr15, hr20, hr25, hr30, hr35, hr40)]
setnames(train, "seq_cut", "seq")

######################################################################################

##Logistic regression

#Use 1st 50 amino acids
start = 1
stop = 50

#AA freq only
x = getFreqsinWindow(train, start, stop)
model=caret::train(x = x , y=train$type, method = "glm", trControl = trainControl(method="cv", number=6)) 
model
confusionMatrix.train(model)
evaluate(model) #Accuracy:  0.9  | PPV:  0.86  | Recall:  0.82  | MCC:  0.77 
save(model, file="past_results/logistic_regression/model_aa.RData")

#Transcriptome only
x = train[, .(hr5, hr10, hr15, hr20, hr25, hr30, hr35, hr40)]
model=caret::train(x = x, y=train$type, method = "glm", trControl = trainControl(method="cv", number=6)) 
evaluate(model) #Accuracy:  0.91  | PPV:  0.86  | Recall:  0.86  | MCC:  0.8
save(model, file="past_results/logistic_regression/model_rna.RData")


#AA freq + transcriptome; L1 regularization
x = cbind(train[, .(hr5, hr10, hr15, hr20, hr25, hr30, hr35, hr40)], getFreqsinWindow(train, start, stop))
model=caret::train(x = x, y=train$type, method = "glmnet", 
                   trControl = trainControl(method="cv", number=6), preProcess=c('center', 'scale')) 
model
confusionMatrix.train(model)
evaluate(model) #Accuracy:  0.91  | PPV:  0.86  | Recall:  0.86  | MCC:  0.8
save(model, file="past_results/logistic_regression/model.RData")

#Feature importance
bet = predict(model$finalModel, type="coef", s = 0.0005099155)[,1]
feature=names(bet)
bet = data.table(bet, feature=feature)
ggplot(bet, aes(x=feature, y=bet, fill = substr(feature, 1,2)=="hr"))+geom_bar(stat="identity")+
  facet_wrap(facet=~substr(feature, 1,2)=="hr", scales="free") + theme(legend.position="none")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +ylab('Coefficient')
