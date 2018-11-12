import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import sys

# Utility function to check the system's center of mass
def center_of_mass(masses, positions):
	total_mass = sum([m for m in masses]) 

	return (1./total_mass) * sum(masses[i]*positions[i] for i in range(0,len(masses)))

# Function to return default initial conditions
def get_initial_conditions(kind='mgr', a=1.):
	if kind == 'pythagorean':
		masses=[3.,4.,5.]
		x0 = np.array([[1.,3.], [-2.,-1.], [1.,-1.]])
		v0 = np.array([[0.,0.], [0.,0.], [0.,0.]])

	elif kind == 'mgr':
		# Vertices of an equilateral triangle
		masses = [1.,1.,1.]
		x_m1 = np.array([0.,0.])
		x_m2 = np.array([a,0.])
		x_m3 = np.array([a/2., a*np.sqrt(3.)/2.])
		
		x0 = np.array([x_m1, x_m2, x_m3])

		# Tangent vectors
		c = np.array([a/2.,a *np.sqrt(3.)/4.])  # center of concentric triangle
		m1c = c - x_m1						    # vec. from center to vertex 1
		v_m1 = np.array([-m1c[1],m1c[0]])	    # flip components & one sign to get perpendicular vec
		v_m1 = 1.05*(1./np.linalg.norm(v_m1)) * v_m1 # normalize vector to avoid drift

		m2c = c - x_m2
		v_m2 = np.array([-m2c[1],m2c[0]])
		v_m2 = 1.05*(1./np.linalg.norm(v_m2)) * v_m2

		m3c = c - x_m3
		v_m3 = np.array([-m3c[1],m3c[0]])
		v_m3 = 1.05*(1./np.linalg.norm(v_m3)) * v_m3

		v0 = -1*np.array([v_m1, v_m2, v_m3])
	else:
		print 'invalid initial conditions parameters'
		sys.exit()

	return x0, v0, masses

# Plotting function
def plot_trajectories(m1x, m1y, m2x, m2y, m3x, m3y, masses, animate=False):
	if animate == True:
		fig = plt.figure()
		ax = plt.axes(xlim=(-5, 5), ylim=(-5,5))
		#ax = plt.axes(xlim=(-0.005, 0.005), ylim=(-0.008,0.008))
		line1, = ax.plot([], [], 'r-',lw=1)
		line2, = ax.plot([], [], 'g-',lw=1)
		line3, = ax.plot([], [], 'b-',lw=1)

		planet1, = ax.plot([], [], 'ro', label='m1', markersize=masses[0])
		planet2, = ax.plot([], [], 'go', label='m2', markersize=masses[1])
		planet3, = ax.plot([], [], 'bo', label='m3', markersize=masses[2])

		def init():
			line1.set_data([], [])
			line2.set_data([], [])
			line3.set_data([], [])
			planet1.set_data([], [])
			planet2.set_data([], [])
			planet3.set_data([], [])
			return line1, line2, line3, planet1, planet2, planet3

		def animate(i):
			#print 'frame',i,'/',len(m1x)
			line1.set_data(m1x[0:i],m1y[0:i])
			planet1.set_data(m1x[i],m1y[i])
			
			line2.set_data(m2x[0:i],m2y[0:i])
			planet2.set_data(m2x[i],m2y[i])
			
			line3.set_data(m3x[0:i],m3y[0:i])
			planet3.set_data(m3x[i],m3y[i])
			return line1, line2, line3, planet1, planet2, planet3

		anim = animation.FuncAnimation(fig, animate, np.arange(1, len(m1x)), init_func=init,
	                                   interval=50, blit=True)
	else:
		#m1 trajectory
		plt.plot(m1x[0], m1y[0], 'rx', markersize=3)
		plt.plot(m1x[len(m1x)-1], m1y[len(m1y)-1], 'ro', markersize=masses[0])
		plt.plot(m1x, m1y, 'r-', label='m1', lw=1)
		#m2 trajectory
		plt.plot(m2x[0], m2y[0], 'gx', markersize=3)
		plt.plot(m2x[len(m2x)-1], m2y[len(m2y)-1], 'go', markersize=masses[1])
		plt.plot(m2x, m2y, 'g-', label='m2', lw=1)
		#m3 trajectory
		plt.plot(m3x[0], m3y[0], 'bx', markersize=3)
		plt.plot(m3x[len(m3x)-1], m3y[len(m3y)-1], 'bo', markersize=masses[2])
		plt.plot(m3x, m3y, 'b-', label='m3', lw=1)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()		
	plt.show()

# Given the masses and positions of 3 bodies at a given time, calculate acceleration of each planet
def accel(masses, x):

	# planet 1 (0, wrt 1 & 2)
	a_m1 = - masses[1] * (x[0] - x[1])/((np.abs(np.linalg.norm(x[0]-x[1])))**3)
	a_m1 = a_m1 - masses[2] * (x[0] - x[2])/((np.abs(np.linalg.norm(x[0]-x[2])))**3)

	# planet 2 (1, wrt 0 & 2)
	a_m2 = - masses[0] * (x[1] - x[0])/((np.abs(np.linalg.norm(x[1]-x[0])))**3)
	a_m2 = a_m2 - masses[2] * (x[1] - x[2])/((np.abs(np.linalg.norm(x[1]-x[2])))**3)

	# planet 3 (2, wrt 0 & 1)
	a_m3 = - masses[0] * (x[2] - x[0])/((np.abs(np.linalg.norm(x[2]-x[0])))**3)
	a_m3 = a_m3 - masses[1] * (x[2] - x[1])/((np.abs(np.linalg.norm(x[2]-x[1])))**3)

	accel = np.zeros((3,2))
	accel[0] = a_m1
	accel[1] = a_m2
	accel[2] = a_m3

	return accel

# 4th and 5th order Runge-Kutta-Fehlberg algorithm
def RKF45(x0, v0, masses, dt):
	a  = np.array([[0.         , 0.           , 0.          , 0.		  , 0.		],
            	   [1./4.      , 0.           , 0.          , 0.		  , 0.   	],
                   [3./32.     , 9./32.       , 0.          , 0.		  , 0.   	],
                   [1932./2197., -7200./2197. , 7296./2197. , 0.		  , 0.   	],
                   [439./216.  , -8.          , 3680./513.  , -845./4104. , 0.      ],
                   [-8./27.    , 2.           , -3544./2565., 1859./4104. , -11./40.]])

	b4 = np.array( [25./216.   , 0.           , 1408./2565. , 2197./4104.  , -1./5. , 0.])

	b5 = np.array( [16./135.   , 0.           , 6656./12825., 28561./56430., -9./50., 2./55.])

	kx1 =  dt*v0
	kv1 =  dt*accel( masses, x0 )

	kx2 =  dt*(v0 + a[1,0]*kv1)
	kv2 =  dt*accel( masses, x0 + a[1,0]*kx1 )

	kx3 =  dt*(v0 + a[2,0]*kv1 + a[2,1]*kv2)
	kv3 =  dt*accel( masses, x0 + a[2,0]*kx1 + a[2,1]*kx2 )

	kx4 =  dt*(v0 + a[3,0]*kv1 + a[3,1]*kv2 + a[3,2]*kv3)
	kv4 =  dt*accel( masses, x0 + a[3,0]*kx1 + a[3,1]*kx2 +a[3,2]*kx3 )

	kx5 =  dt*(v0 + a[4,0]*kv1 + a[4,1]*kv2 + a[4,2]*kv3 + a[4,3]*kv4)
	kv5 =  dt*accel( masses, x0 + a[4,0]*kx1 + a[4,1]*kx2 + a[4,2]*kx3 + a[4,3]*kx4 )

	kx6 =  dt*(v0 + a[5,0]*kv1 + a[5,1]*kv2 + a[5,2]*kv3 + a[5,3]*kv4 + a[5,4]*kv5)
	kv6 =  dt*accel( masses, x0 + a[5,0]*kx1 + a[5,1]*kx2 + a[5,2]*kx3 + a[5,3]*kx4+ a[5,4]*kx5 )

	x4 = x0 + (b4[0]*kx1 + b4[1]*kx2 + b4[2]*kx3 + b4[3]*kx4 + b4[4]*kx5)
	v4 = v0 + (b4[0]*kv1 + b4[1]*kv2 + b4[2]*kv3 + b4[3]*kv4 + b4[4]*kv5)

	x5 = x0 + (b5[0]*kx1 + b5[1]*kx2 + b5[2]*kx3 + b5[3]*kx4 + b5[4]*kx5 + b5[5]*kx6)
	v5 = v0 + (b5[0]*kv1 + b5[1]*kv2 + b5[2]*kv3 + b5[3]*kv4 + b5[4]*kv5 + b5[5]*kv6)

	return x4, v4, x5, v5

# Function that calls RKF45 scheme iteratively with an adapting timestep
def solver(x0, v0, dt, t, T, masses, max_epsilon): 

	# initialize arrays to store all accepted 5th order solutions
	x_array = x0
	v_array = v0

	# initialize time array for plotting
	t_array = t

	# initialize vars to hold current values of x and v
	x = x0
	v = v0

	epsilon = 0.

	while t < T:
		print dt
		x4_new, v4_new, x5_new, v5_new = RKF45(x, v, masses, dt)

		# Check q and adapt dt in any case
		epsilon_x  = np.abs(x4_new-x5_new)
		epsilon_v  = np.abs(v4_new-v5_new)
		epsilon_xv = np.array([epsilon_x, epsilon_v])
		epsilon    = np.amax(epsilon_xv)

		q = (max_epsilon/(2.*epsilon))**(0.2)

		if q < 1.:
			dt = q*dt
			continue
		else:
			x = x5_new
			v = v5_new

			x_array = np.vstack((x_array, x5_new))
			v_array = np.vstack((v_array, v5_new))
			t_array = np.append(t_array, t)

		t = t + dt
		dt = q*dt

	# Create x,y arrays from final data for easy plotting
	m1x = []; m1y = []; m2x = []; m2y = []; m3x = []; m3y = []

	rows = len(x_array) # num of rows is the total number of accepted xs & vs we have for the system
	for i in range(0, rows/3):
		m1x.append(x_array[3*i][0])
		m1y.append(x_array[3*i][1])

		m2x.append(x_array[3*i+1][0])
		m2y.append(x_array[3*i+1][1])

		m3x.append(x_array[3*i+2][0])
		m3y.append(x_array[3*i+2][1])


	return np.array(m1x), np.array(m1y), np.array(m2x), np.array(m2y), np.array(m3x), np.array(m3y), np.array(t_array)

if __name__ == '__main__':

	# Initial conditions
	x0, v0, masses = get_initial_conditions('pythagorean')

	dt = 0.01
	t = 0.
	T = 20.
	max_epsilon = 1E-4

	#### Calculate and plot trajectory of three bodies from t to T
	m1x_full, m1y_full, m2x_full, m2y_full, m3x_full, m3y_full, t_array = solver(x0, v0, dt, t, T, masses, max_epsilon)
	plot_trajectories(m1x_full, m1y_full, m2x_full, m2y_full, m3x_full, m3y_full, masses, animate=True)
