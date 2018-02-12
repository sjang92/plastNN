import sys

def parseData():
    print 'startParsing'
    file = 'sequence.txt'
    proteins = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            if line[0] == 'id':
            	continue
            if len(line) != 2:
                break
            p = Protein(line[0], +1)
            p.seq =line[1]
            proteins.append(p)
            print p.id
    return proteins

def writeFile(proteins):
    hydroPH7File = 'hydroPH7.txt'
    chargeFile = 'charge.txt'
    MWFile = 'MW.txt'
    pKaFile = 'pKa.txt'
    with open(hydroPH7File, 'w') as fout:
        for protein in proteins:
            fout.write(str(protein.id)+' '+protein.hydroPH7+'\n')
    with open(chargeFile, 'w') as fout:
        for protein in proteins:
            fout.write(str(protein.id)+' '+protein.charge+'\n')
    with open(MWFile, 'w') as fout:
        for protein in proteins:
            fout.write(str(protein.id)+' '+protein.MW+'\n')
    with open(pKaFile, 'w') as fout:
        for protein in proteins:
            fout.write(str(protein.id)+' '+protein.pKa+'\n')

class Protein:
    def __init__(self, id, label):
        self.id = id
        self.label = label
        self.hydroPH7 = None
        self.charge = None
        self.MW = None
        self.pKa = None
        self.seq = None

def addFeature(proteins):
	dicHydroPH7 = {'F':100, 'I':99, 'W':97, 'L':97, 'V':76, 'M':74, 'Y':63, 'C':49, 'A':41, 'T':13, 'H':8, 'G':0, 'S':-5, 'Q':-10, 'R':-14, 'K':-23, 'N':-28, 'E':-31, 'P':-46, 'D':-55}
	dicCharge = {'F':0, 'I':0, 'W':0, 'L':0, 'V':0, 'M':0, 'Y':0, 'C':0, 'A':0, 'T':0, 'H':1, 'G':0, 'S':0, 'Q':0, 'R':1, 'K':1, 'N':0, 'E':-1, 'P':0, 'D':-1}
	dicMW = {'A':89.10, 'R':174.20, 'N':132.12, 'D':133.11, 'C':121.16, 'E':147.13, 'Q':146.15, 'G':75.07, 'H':155.16, 'I':131.18, 'L':131.18, 'K':146.19, 'M':149.21, 'F':165.19, 'P':115.13, 'S':105.09, 'T':119.12, 'W':204.23, 'Y':181.19, 'V':117.15}
	dicPKa = {'A':2.34, 'R':2.17, 'N':2.02, 'D':1.88, 'C':1.96, 'E':2.19, 'Q':2.17, 'G':2.34, 'H':1.82, 'I':2.36, 'L':2.36, 'K':2.18, 'M':2.28, 'F':1.83, 'P':1.99, 'S':2.21, 'T':2.09, 'W':2.83, 'Y':2.20, 'V':2.32}
	for i in range(len(proteins)):
		strHydro = ''
		strCharge = ''
		strMW = ''
		strPKa = ''
		protein = proteins[i]
		for j in range(len(protein.seq)):
			strHydro += str(dicHydroPH7[protein.seq[j]])
			strHydro += ' '
			strCharge += str(dicCharge[protein.seq[j]])
			strCharge += ' '
			strMW += str(dicMW[protein.seq[j]])
			strMW += ' '
			strPKa += str(dicPKa[protein.seq[j]])
			strPKa += ' '
		proteins[i].hydroPH7 = strHydro
		proteins[i].charge = strCharge
		proteins[i].MW = strMW
		proteins[i].pKa = strPKa  

def main(args):
    proteins = parseData()
    addFeature(proteins)
    writeFile(proteins)
if __name__=='__main__':
    main(sys.argv)
        