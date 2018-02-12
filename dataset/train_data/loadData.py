import sys

def parseData():
    print 'startParsing'
    posFile = 'positive.txt'
    negFile = 'negative.txt'
    proteins = []
    with open(posFile, 'r') as f:
        for line in f:
            line =line.strip()
            line = line.split()
            if len(line) != 2:
                break
            p = Protein(line[0], +1)
            p.seq =line[1]
            proteins.append(p)
            print p.id
    #read negative proteins
    with open(negFile, 'r') as f:
        for line in f:
            line =line.strip()
            line = line.split()
            if len(line) != 2:
                break
            p = Protein(line[0], -1)
            p.seq =line[1]
            proteins.append(p)
            print p.id
    return proteins
def writeFile(proteins):
    sequenceFile = 'sequence.txt'
    labelFile = 'label.txt'
    with open(sequenceFile, 'w') as fout:
        for protein in proteins:
            fout.write(str(protein.id)+' '+protein.seq+'\n')
    with open(labelFile, 'w') as fout2:
        for protein in proteins:
            fout2.write(str(protein.id)+' '+str(protein.label)+'\n')
class Protein:
    def __init__(self, id, label):
        self.id = id
        self.label = label
        self.annotation = None
        self.seq = None
        
def main(args):
    proteins = parseData()
    writeFile(proteins)
if __name__=='__main__':
    main(sys.argv)
        

