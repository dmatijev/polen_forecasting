from matplotlib import pyplot as plt

def predictionsGraph(realValues, predictedValues):
    plt.clf()
    plt.plot(range(1,len(realValues)+1),realValues, label="stvarna", color="red")
    plt.plot(range(1,len(predictedValues)+1),predictedValues, label="predviđena", color="blue")
    plt.title('Prikaz stvarne i predviđene PRAM po danima')
    plt.legend()
    plt.show()
    
def predictionsGraphScatter(realValues, predictedValues):
    plt.clf()
    plt.scatter(range(1,len(realValues)+1),realValues, label="stvarna", color="red", marker='o')
    plt.scatter(range(1,len(predictedValues)+1),predictedValues, label="predviđena", color="blue", marker='s')
    plt.title('Prikaz stvarne i predviđene PRAM po danima')
    plt.legend()
    plt.show()