from matplotlib import pyplot as plt

def predictionsGraph(realValues, predictedValues, oldPredictedValues=None, pollenCalendar=None, ):
    plt.clf()
    plt.plot(range(1,len(realValues)+1),realValues, label="stvarna", color="red")
    plt.plot(range(1,len(predictedValues)+1),predictedValues, label="LSTM", color="blue")
    if pollenCalendar is not None:
        plt.plot(range(1,len(pollenCalendar)+1),pollenCalendar, label="kalendar", color="orange")
    if oldPredictedValues is not None:
        plt.plot(range(1,len(oldPredictedValues)+1),oldPredictedValues, label="patterns", color="green")
    plt.title('Prikaz stvarne i predviđene PRAM po danima')
    plt.legend()
    plt.show()
    
def predictionsGraphScatter(realValues, predictedValues, oldPredictedValues=None, pollenCalendar=None, ):
    plt.clf()
    plt.scatter(range(1,len(realValues)+1),realValues, label="stvarna", color="red", marker='o')
    plt.scatter(range(1,len(predictedValues)+1),predictedValues, label="LSTM", color="blue", marker='s')
    if pollenCalendar is not None:
        plt.scatter(range(1,len(pollenCalendar)+1),pollenCalendar, label="kalendar", color="orange", marker='D')
    if oldPredictedValues is not None:
        plt.scatter(range(1,len(oldPredictedValues)+1),oldPredictedValues, label="patterns", color="green", marker='*')
    plt.title('Prikaz stvarne i predviđene PRAM po danima')
    plt.legend()
    plt.show()