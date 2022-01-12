from matplotlib import pyplot as plt
    
def predictionsGraph(values, labels):
    """(list of lists, list of strings) -> NoneType
    values - lista kojoj su elementi liste s vrijednostima koje treba nacrtati; npr. prva lista su
    stvarne vrijednosti, druga lista su predikcije dobivene LSTM-om itd.
    labels - nazivi za pojedinu listu vrijednosti koji će se ispisati na grafu, npr. stvarna, LSTM, itd."""
    colors = ["red", "blue", "green", "orange", "purple", "yellow", "brown", "gray"]
    plt.clf()
    for i in range(len(labels)):
        plt.plot(range(1,len(values[i])+1),values[i], label=labels[i], color=colors[i])
    plt.title('Prikaz stvarne i predviđene PRAM po danima')
    plt.legend()
    plt.show()
    
def predictionsGraphScatter(values, labels):
    """(list of lists, list of strings) -> NoneType
    values - lista kojoj su elementi liste s vrijednostima koje treba nacrtati; npr. prva lista su
    stvarne vrijednosti, druga lista su predikcije dobivene LSTM-om itd.
    labels - nazivi za pojedinu listu vrijednosti koji će se ispisati na grafu, npr. stvarna, LSTM, itd."""
    colors = ["red", "blue", "green", "orange", "purple", "yellow", "brown", "gray"]
    markers = ['o', 's', '*', 'D', 'v', 'P', 'X', 'H']
    plt.clf()
    for i in range(len(labels)):
        plt.scatter(range(1,len(values[i])+1),values[i], label=labels[i], color=colors[i], marker=markers[i])
    plt.title('Prikaz stvarne i predviđene PRAM po danima')
    plt.legend()
    plt.show()