import argparse 

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def setup():
	""" Simple 3 Mode Controlled Setup """

	## Observations (t = 1 to 8)
	x_obs = np.tile(np.linspace(1, 8, num=8, endpoint=True), (3, 1))
	y_obs = np.zeros((3, 8))
	
	## Real Predictions (t = 9 to 16)
	x_pred = np.tile(np.linspace(9, 16, num=8, endpoint=True), (3, 1))
	y_pred = np.zeros((3, 8))

	## 1st Mode (Straight Line): y = 0
	## No edits required 

	## 2nd Mode (Up Slant Line)
	y_pred[1, :] = 1*np.linspace(1, 8, num=8, endpoint=True)
	## 3rd Mode (Down Slant Line)
	y_pred[2, :] = -1*np.linspace(1, 8, num=8, endpoint=True)

	x = np.concatenate((x_obs, x_pred), axis=1)
	y = np.concatenate((y_obs, y_pred), axis=1)
	f1 = interp1d(x[0, :], y[0, :], kind='cubic')
	f2 = interp1d(x[1, :], y[1, :], kind='cubic')
	f3 = interp1d(x[2, :], y[2, :], kind='cubic')

	return (f1, f2, f3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default="single_traj",
                        help='output file name')
    parser.add_argument('--num_traj', default=500, type=int,
                        help='number of samples')
    parser.add_argument('--num_modes', default=1, type=int,
                        help='number of modes')
    args = parser.parse_args()

    print("HERE")
    ## Get setup functions
    fs = setup()
    filename = args.filename + '.txt'
    num = args.num_traj
    modes = args.num_modes 
    prob = [1.00, 0.00, 0.00]
    if modes == 3:
        prob = [0.34, 0.33, 0.33]

    x = np.linspace(1, 16, num=16, endpoint=True)
    with open(filename, 'a') as the_file:
        frame_id = 0
        for i in range(num):
            ch = np.random.choice(3, 1, p=prob)
            f = fs[ch[0]]
            print(ch)
            # n = np.random.normal(scale=0.02, size=16)
            y = f(x)
            y = np.around(y, 3)
            x = np.around(x, 3)
            plt.plot(x, y, 'r')
            # ax.legend()
            plt.ylim((-10, 10))
            plt.xlim((0, 20))
            for j in range(16):
                frame_id += 1
                the_file.write(str(frame_id) + '\t' + str(i) + '\t' + str(x[j]) + '\t' + str(y[j]) + '\n')
    plt.show()

if __name__ == '__main__':
    main()