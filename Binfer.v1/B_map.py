import math
import numba as nb
import numpy as np
import scipy.integrate as integrate
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("default", category=DeprecationWarning)
np.set_printoptions(linewidth=np.inf)

############################################### integral functions ###############################################

@nb.njit
def gamma_BS_RM(s, s_mean, s_shape, h, r):
	return (s**(s_shape-1) * np.exp(-s * (s_shape / s_mean)) * (s_shape / s_mean)**s_shape) / math.gamma(s_shape) * (h*s * (1 + (-2*r + h*(-1 + 2*r)*s)**2) / (r + h*s - h*r*s)**2)

@nb.njit
def gamma_BS_selfing(s, s_mean, s_shape, h, r, alpha):
	return (s**(s_shape-1) * np.exp(-s * (s_shape / s_mean)) * (s_shape / s_mean)**s_shape) / math.gamma(s_shape) * (s**2*(2 - alpha)*(alpha - 2*h*(alpha - 1))**2*(2*r + 1)**2/((2*r*(-alpha*s + alpha + 2*h*s*(alpha - 1) - 1) + s*(-alpha + 2*h*(alpha - 1)))**2*(-2*alpha*h*s + alpha*s + 2*h*s)))

@nb.njit
def gamma_BS_RM_within(s, s_mean, s_shape, h, rf):
	return (s**(s_shape-1) * np.exp(-s * (s_shape / s_mean)) * (s_shape / s_mean)**s_shape) / math.gamma(s_shape) * ((rf - (rf * s*h) + s*h * math.log(s*h / (rf + s*h - (rf * s*h)))) / (rf**2 * (s*h - 1)**2))

@nb.njit
def gamma_BS_selfing_within(s, s_mean, s_shape, h, rf, alpha):
	return (s**(s_shape-1) * np.exp(-s * (s_shape / s_mean)) * (s_shape / s_mean)**s_shape) / math.gamma(s_shape) * ((-2 + alpha)**2 * (-2*rf*(-1 + alpha)*(2 + 2*h*s*(-1 + alpha) - (1 + s)*alpha) + s*(2*h*(-1 + alpha) - alpha)*(-2 + alpha) * math.log(-((s*(2*h*(-1 + alpha) - alpha)*(-2 + alpha))/(-s*(2*h*(-1 + alpha) - alpha)*(-2 + alpha) + 2*rf*(-1 + alpha)*(2 + 2*h*s*(-1 + alpha) - (1 + s)*alpha))))))/(4*rf**2 *(-1 + alpha)**2 *(2 + 2*h*s*(-1 + alpha) - (1 + s)*alpha)**2)

@nb.njit
def gamma_ND(s, s_mean, s_shape, t):
    return (s**(s_shape-1) * np.exp(-s * (s_shape / s_mean)) * (s_shape / s_mean)**s_shape) / math.gamma(s_shape) * (1 - (1 - np.exp(-t*s))**2)

@nb.njit
def just_gamma(s, s_mean, s_shape):
    return (s**(s_shape-1) * np.exp(-s * (s_shape / s_mean)) * (s_shape / s_mean)**s_shape) / math.gamma(s_shape)

def BS_integral(s_shape, s_mean, h, r, min_s):
	return integrate.quad(lambda s: gamma_BS_RM(s, s_mean, s_shape, h, r), min_s, 1, epsrel=1e-4)[0]

def BS_integral_selfing(s_shape, s_mean, h, r, alpha, min_s):
	return integrate.quad(lambda s: gamma_BS_selfing(s, s_mean, s_shape, h, r, alpha), min_s, 1, epsrel=1e-4)[0]

def BS_within_integral(s_shape, s_mean, h, rf, min_s):
	return integrate.quad(lambda s: gamma_BS_RM_within(s, s_mean, s_shape, h, rf), min_s, 1, epsrel=1e-4)[0]

def BS_within_integral_selfing(s_shape, s_mean, h, rf, alpha, min_s):
	return integrate.quad(lambda s: gamma_BS_selfing_within(s, s_mean, s_shape, h, rf, alpha), min_s, 1, epsrel=1e-4)[0]

def gamma_ND_integral(s_shape, s_mean, t, min_s):
    return integrate.quad(lambda s: gamma_ND(s, s_mean, s_shape, t), min_s, 1, epsrel=1e-4)[0] / integrate.quad(lambda s: just_gamma(s, s_mean, s_shape), min_s, 1)[0] # get mean 


############################################ ND step functions ##########################################

def get_ND_step(s_shape, s_mean, min_s):
	gamma_ND_points = [1]
	for gen in range(10, 100_000, 10):
		point = gamma_ND_integral(s_shape, s_mean, gen, min_s)
		gamma_ND_points.append(point)
		if point < 1e-2:
			break
	av = (gamma_ND_points[0]*5 + sum(gamma_ND_points[1:-1])*10 + gamma_ND_points[-1]*5) / ((len(gamma_ND_points) * 10) - 10)
	return av*gen

def get_ND_two_step(s_shape, s_mean, min_s):
	keep_going = True
	half_way_found = False
	gen = 0
	delta = 10
	gamma_ND_points = [1]

	while keep_going:
		gen += delta
		point = gamma_ND_integral(s_shape, s_mean, gen, min_s)
		gamma_ND_points.append(point)
		if point < 0.5 and not half_way_found:
			half_way_idx = len(gamma_ND_points) - 1
			half_way_gen = gen
			step_1_av = ((gamma_ND_points[0]*5) + (gamma_ND_points[-1]*5) + (sum(gamma_ND_points[1:-1])*10)) / ((len(gamma_ND_points) * 10) - 10)
			step_1 = (2*step_1_av - 1) * gen
			half_way_found = True
			delta = 50
		if point < 1e-2:
			step_2_av = ((gamma_ND_points[half_way_idx]*25) + (gamma_ND_points[-1]*25) + (sum(gamma_ND_points[half_way_idx+1:-1])*50)) / ((len(gamma_ND_points[half_way_idx:None]) * 50) - 50)
			step_2 = step_1 + ((2 * step_2_av) * (gen - half_way_gen))
			keep_going = False
	return step_1, step_2

############################################ interpolation functions ##########################################

########## arrays used for interpolation ###############

r_inter = np.array([5e-5, 1e-5, 
	2.5e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 
	2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 
	2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 
	2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 0.1, 
	0.2, 0.3, 0.4, 0.5])

r_inter_within = np.array([5e-10, 7.5e-10, 1e-9, 2.5e-9, 5e-9, 7.5e-9, 
	1e-8, 2.5e-8, 5e-8, 7.5e-8, 1e-7]) * 10_000

@nb.njit
def linear_interpolation(rf, BS_values, inter_values):

	out_array = np.zeros(rf.shape, dtype=float)

	for i, r in np.ndenumerate(rf):

		if r == 0.5:
			out_array[i] = BS_values[-1]

		elif r == 0:
			out_array[i] = 0

		else:

			x = r
			
			x1 = np.searchsorted(inter_values, x, side='left')
			x0 = x1 - 1
			xd = (x - inter_values[x0]) / (inter_values[x1] - inter_values[x0])
			y0 = BS_values[x0]
			y1 = BS_values[x1]
			y = y0 + (xd * (y1 - y0))
	
			out_array[i] = y

	return out_array


############################################### B-map ###############################################

def get_B_map(s_shape, s_mean, u, alpha, h, min_s, r_distance_array, r_array, exon_array, window_size):

	if alpha == None:

		try:
			BS_values_between = np.array([BS_integral(s_shape, s_mean, h, r, min_s) for r in r_inter])
			BS_values_within = np.array([BS_within_integral(s_shape, s_mean, h, rf, min_s) for rf in r_inter_within])
		except:
			return None

	else:

		try:
			BS_values_between = np.array([BS_integral_selfing(s_shape, s_mean, h, r, alpha, min_s) for r in r_inter])
			BS_values_within = np.array([BS_within_integral_selfing(s_shape, s_mean, h, rf, alpha, min_s) for rf in r_inter_within])
		except:
			return None

	interpolated_BS_terms = linear_interpolation(r_distance_array, BS_values_between, r_inter)
	interpolated_BS_terms_within = linear_interpolation(r_array * window_size, BS_values_within, r_inter_within)

	BS_array_between = np.multiply(exon_array, interpolated_BS_terms * u)
	BS_array_within = np.multiply(exon_array, interpolated_BS_terms_within * u * 2)

	B_map = np.array([np.exp(-1 * j) for j in np.add(np.sum(BS_array_between, 1), BS_array_within)])

	return B_map


def get_B_map_mp(s_shape, s_mean, u, alpha, h, min_s, r_distance_array, r_array, exon_array, window_size, processes):

	if alpha == None:

		try:
			# between could be split up
			BS_values_between = np.array([BS_integral(s_shape, s_mean, h, r, min_s) for r in r_inter])
			BS_values_within = np.array([BS_within_integral(s_shape, s_mean, h, rf, min_s) for rf in r_inter_within])
		except:
			return None

	else:

		try:
			BS_values_between = np.array([BS_integral_selfing(s_shape, s_mean, h, r, alpha, min_s) for r in r_inter])
			BS_values_within = np.array([BS_within_integral_selfing(s_shape, s_mean, h, rf, alpha, min_s) for rf in r_inter_within])
		except:
			return None

	# both arrays could be split up then stuck back together (?)
	interpolated_BS_terms = linear_interpolation(r_distance_array, BS_values_between, r_inter)
	interpolated_BS_terms_within = linear_interpolation(r_array * window_size, BS_values_within, r_inter_within)

	BS_array_between = np.multiply(exon_array, interpolated_BS_terms * u)
	BS_array_within = np.multiply(exon_array, interpolated_BS_terms_within * u * 2)

	B_map = np.array([np.exp(-1 * j) for j in np.add(np.sum(BS_array_between, 1), BS_array_within)])

	return B_map
