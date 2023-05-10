import numpy as np
import matplotlib.pyplot as plt

from IPython import display

plt.ion()

scores = []
mean_scores = []

def plot(score):

	# Clear the display and current figure
	display.clear_output(wait = True)
	display.display(plt.gcf())
	plt.clf()

	# Set the title and labesl
	plt.title('Training...')
	plt.xlabel('Number of Games')
	plt.ylabel('Score')

	# Add the score to the list and calculate the mean
	scores.append(score)
	mean_scores.append(np.mean(scores))

	# Plot the data
	plt.plot(scores)
	plt.plot(mean_scores)

	# Add the legend and show the plot
	plt.ylim(ymin=0)
	plt.text(len(scores)-1, scores[-1], str(scores[-1]))
	plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
	plt.show(block=False)
	plt.pause(.1)