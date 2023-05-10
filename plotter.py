import numpy as np
import matplotlib.pyplot as plt

from IPython import display

plt.ion()

scores = []
mean_scores = []

def plot(score):

	display.clear_output()
	display.display(plt.gcf())

	plt.clf()
	plt.title('Training results')
	plt.xlabel('Number of Games')
	plt.ylabel('Score')

	scores.append(score)
	mean_scores.append(np.mean(scores))

	plt.plot(scores)
	plt.plot(mean_scores)

	plt.show()
	plt.pause(.1)