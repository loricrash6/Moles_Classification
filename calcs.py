import numpy as np
if __name__ == '__main__':
	
	#BORDER ANALYSIS
	infile = open("perimeters.txt", 'r')

	numbers = [line for line in infile.readlines()]
	infile.close()

	#array of values for the border indentation of each mole
	ns=[]
	for it in range(len(numbers)):
		n=float(numbers[it].split("\n")[0])
		ns.append(n)
		#print(n[it])

	#splitting into classes
	low=ns[0:11]
	med=ns[11:27]
	mel=ns[27:]
	print(low)
	print('\n')
	print(med)
	print('\n')
	print(mel)

	print("LOW RISK - mean: "+str(round(np.mean(low),3))+", stdev: "+str(round(np.std(low),3))+"\n")
	print("MEDIUM RISK - mean: "+str(round(np.mean(med),3))+", stdev: "+str(round(np.std(med),3))+"\n")
	print("MELANOMA - mean: "+str(round(np.mean(mel),3))+", stdev: "+str(round(np.std(mel),3))+"\n")

	#SYMMETRY ANALYSIS
	infile = open("symmetries.txt", 'r')

	numbers = [line for line in infile.readlines()]
	infile.close()
	
	ns=[]
	for it in range(len(numbers)):
		n=float(numbers[it].split("\n")[0])
		ns.append(n)
		#print(n[it])
	low=ns[0:11]
	med=ns[11:27]
	mel=ns[27:]
	print(low)
	print('\n')
	print(med)
	print('\n')
	print(mel)

	print("LOW RISK - mean: "+str(round(np.mean(low),3))+", stdev: "+str(round(np.std(low),3))+"\n")
	print("MEDIUM RISK - mean: "+str(round(np.mean(med),3))+", stdev: "+str(round(np.std(med),3))+"\n")
	print("MELANOMA - mean: "+str(round(np.mean(mel),3))+", stdev: "+str(round(np.std(mel),3))+"\n")