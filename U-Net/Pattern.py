import numpy as np
import matplotlib.pyplot as plt

width = 400
height = 400

x = np.linspace(0, 40 * np.pi, width)
y = np.linspace(0, 40 * np.pi, height)

X, Y = np.meshgrid(x, y)

horizontal_stripes = 0.5 * (1 + np.sin(X))
vertical_stripes = 0.5 * (1 + np.sin(Y))

chessboard = (horizontal_stripes + vertical_stripes) / 2

plt.imshow(chessboard, cmap='binary', interpolation='nearest')
plt.axis('off') 

plt.savefig('chessboard.png', bbox_inches='tight', pad_inches=0)


plt.show()
