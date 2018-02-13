#Set working directory
setwd("~/Documents/GitHub/plastNN/")

#Load libraries
library(data.table)
library(seqinr)
library(readxl)

#1: Read positive, negative and test files
pos_set = as.data.table(read_xlsx("data/raw_data/120417_UpdatedTrainingSets.xlsx", sheet = 1))
neg_set = as.data.table(read_xlsx("data/raw_data/120417_UpdatedTrainingSets.xlsx", sheet = 2))
allsp_set = as.data.table(read_xlsx("data/raw_data/120417_UpdatedTrainingSets.xlsx", sheet = 3))
test_set = allsp_set[!(`Gene ID` %in% c(pos_set$`Gene ID`, neg_set$`Gene ID`))]

#Read fasta file
proteome = read.fasta("data/raw_data/ApicoplastMachineLearning_PlasmoDB32_Pf3D7Proteome.txt", 
                      seqtype = "AA", as.string = T, strip.desc=T, set.attributes=F)
proteome = data.table(id = names(proteome), seq = unlist(proteome))

#Check for presence in fasta file
pos_set = pos_set[`Gene ID` %in% proteome$id]
neg_set = neg_set[`Gene ID` %in% proteome$id]
test_set = test_set[`Gene ID` %in% proteome$id]

#Get IDs
pos_id = pos_set$`Gene ID`
neg_id = neg_set$`Gene ID`
test_id = test_set$`Gene ID`
  
#Merge with sequences
pos = merge(proteome, pos_set[, .(id=`Gene ID`, first_aa_of_tp=`Cleavage Position`)], by="id")
neg = merge(proteome, neg_set[, .(id=`Gene ID`, first_aa_of_tp=`Cleavage Position`)], by="id")
test = merge(proteome, test_set[, .(id=`Gene ID`, first_aa_of_tp=`Cleavage Position`)], by="id")

#Add length
pos[, len:=nchar(seq), by = 1:nrow(pos)]
neg[, len:=nchar(seq), by = 1:nrow(neg)]
test[, len:=nchar(seq), by = 1:nrow(test)]

#Check remaining sequence lengths
pos[,min(len-first_aa_of_tp+1)] #121
neg[,min(len-first_aa_of_tp+1)] #64
test[,min(len-first_aa_of_tp+1)] #5

#Write sequence files
write.table(neg, file = "negative.txt", quote=F, row.names = F)
write.table(pos, file = "positive.txt", quote=F, row.names = F)
write.table(test, file = "test.txt", quote=F, row.names = F)

#Cut IDs to match RNA
pos_id_cut = substr(pos_id,1,13)
neg_id_cut = substr(neg_id,1,13)
test_id_cut = gsub("\\.1$|\\.2$|\\.3$|\\:mRNA$", "", test_id)

#Get scaled values for RNA
rna = fread("data/raw_data/transcriptome_bartfai.txt")
rna = rna[, c(1, 15:22), with=F]
setnames(rna, c("id", "hr5", "hr10", "hr15", "hr20", "hr25", "hr30", "hr35", "hr40"))
rna = unique(rna)
rna = rna[id %in% c(pos_id_cut, neg_id_cut, test_id_cut)]
write.table(rna, file = "rna.txt", quote=F, row.names = F)

#Get RNA intervals for each set
pos_rna_intervals = merge(data.table(id = pos_id, id_cut = pos_id_cut), rna, by.x = "id_cut", by.y = "id")
neg_rna_intervals = merge(data.table(id = neg_id, id_cut = neg_id_cut), rna, by.x = "id_cut", by.y = "id")
test_rna_intervals = merge(data.table(id = test_id, id_cut = test_id_cut), rna, by.x = "id_cut", by.y = "id")

write.table(pos_rna_intervals, file = "pos_rna_intervals.txt", quote=F, row.names = F)
write.table(neg_rna_intervals, file = "neg_rna_intervals.txt", quote=F, row.names = F)
write.table(test_rna_intervals, file = "test_rna_intervals.txt", quote=F, row.names = F)

#List TP start positions
pos_tp = pos[, .(id, first_aa_of_tp)]
neg_tp = neg[, .(id, first_aa_of_tp)]
test_tp = test[, .(id, first_aa_of_tp)]

write.table(pos_tp, file = "pos_tp.txt", quote=F, row.names = F)
write.table(neg_tp, file = "neg_tp.txt", quote=F, row.names = F)
write.table(test_tp, file = "test_tp.txt", quote=F, row.names = F)
