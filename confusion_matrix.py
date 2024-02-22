import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from hw2 import *

def main():
    results = {1: [], 2: [], 3: [], 4: [], 5: []}
    for i in range(1, 6):
        print("STARTING...", i)
        for j in range(50):
            res = count_fingers()
            results[i].append(res)
            print(j)
    print(results)

    actual = [1] * 50 + [2] * 50 + [3] * 50 + [4] * 50 + [5] * 50
    detected = results[1] + results[2] + results[3] + results[4] + results[5]

    print(detected)

    cm = confusion_matrix(actual, detected)
    total_per_class = np.sum(cm, axis=1)
    percentages = np.around((cm.astype('float') / total_per_class[:, np.newaxis]) * 100, decimals=1)

    # Create labels for the heatmap
    labels = []
    for i in range(5):
        for j in range(5):
            labels.append(f'{percentages[i][j]}%\n({cm[i][j]})')

    # Reshape the labels to match the shape of the confusion matrix for plotting
    labels = np.array(labels).reshape(5, 5)

    # Plot the heatmap
    sns.heatmap(cm, annot=labels, fmt='', cmap='OrRd')

    # Set the x and y axis labels to range from 1 to 5
    plt.xticks(np.arange(5) + 0.5, labels=np.arange(1, 6))
    plt.yticks(np.arange(5) + 0.5, labels=np.arange(1, 6))

    plt.xlabel('Detected')
    plt.ylabel('Actual')
    plt.show()

    accuracy = accuracy_score(actual, detected)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
