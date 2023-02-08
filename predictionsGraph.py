from matplotlib import pyplot as plt
    
def predictionsGraph(values, labels, target):
    """(list of lists, list of strings) -> NoneType
    values - list whose elements are lists with values that need to be plotted; e.g. first list are the real values,
    second list are LSTM obtained predictions etc.
    labels - labels for a given list of values that will be shown on plot, e.g. real, LSTM etc."""
    colors = ["red", "blue", "green", "orange", "purple", "yellow", "brown", "gray"]
    plt.clf()
    for i in range(len(labels)):
        plt.plot(range(1,len(values[i])+1),values[i], label=labels[i], color=colors[i])
    plt.title(f'Real and predicted {target} by days')
    plt.legend()
    plt.show()
    
def predictionsGraphScatter(values, labels, target):
    """(list of lists, list of strings) -> NoneType
    values - list whose elements are lists with values that need to be plotted; e.g. first list are the real values,
    second list are LSTM obtained predictions etc.
    labels - labels for a given list of values that will be shown on plot, e.g. real, LSTM etc."""
    colors = ["red", "blue", "green", "orange", "purple", "yellow", "brown", "gray"]
    markers = ['o', 's', '*', 'D', 'v', 'P', 'X', 'H']
    plt.clf()
    for i in range(len(labels)):
        plt.scatter(range(1,len(values[i])+1),values[i], label=labels[i], color=colors[i], marker=markers[i])
    plt.title(f'Real and predicted {target} by days')
    plt.legend()
    plt.show()