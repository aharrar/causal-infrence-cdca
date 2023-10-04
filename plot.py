import matplotlib.pyplot as plt


def plot1(lst, title):
    plt.plot(lst)
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.title(title)
    plt.show()


def plot2(lst1, lst2, title1, title2):
    plt.plot(lst1, label=title1)
    plt.plot(lst2, label=title2)

    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title('Comparison of {} and {}'.format(title1, title2))
    plt.legend()
    plt.show()
