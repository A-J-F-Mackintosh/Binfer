import sys
import math
import numpy as np
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("default", category=DeprecationWarning)
np.set_printoptions(linewidth=np.inf)

import contextlib
import multiprocessing

########################################### multiprocessing functions ###########################################

@contextlib.contextmanager
def poolcontext(*args, **kwargs):
	pool = multiprocessing.Pool(*args, **kwargs)
	yield pool
	pool.terminate()

def chunks(lst, n):
	for i in range(0, len(lst), n):
		yield lst[i:i + n]

######################################################## crs2_lm #############################################


def generate_point(dist, mean, sd, lower, upper):

	if dist == "n":
		point = np.random.normal(mean, sd)
		if lower < point < upper:
			return point
		else:
			try:
				return generate_point(dist, mean, sd, lower, upper)
			except:
				return generate_point("u", mean, sd, lower, upper)

	elif dist == "e":
		point = lower + np.random.exponential(mean - lower)
		if point < upper:
			return point
		else:
			try:
				return generate_point(dist, mean, sd, lower, upper)
			except:
				return generate_point("u", mean, sd, lower, upper)

	elif dist == "u":
		point = lower + (np.random.uniform() * (upper - lower))
		return point

	elif dist == "lu":
		point = 10**(math.log10(lower) + (np.random.uniform() * (math.log10(upper) - math.log10(lower))))
		return point

	elif dist == "ln":
		point = np.random.lognormal(mean=np.log(mean), sigma=np.sqrt(np.log(1 + (sd**2 / mean**2))))
		if lower < point < upper:
			return point
		else:
			try:
				return generate_point(dist, mean, sd, lower, upper)
			except:
				return generate_point("u", mean, sd, lower, upper)
				
	else:
		sys.exit("only normal (n), lognormal (ln), exponential (e), uniform (u) and loguniform (lu) distributions are supported")


def generate_proposal(pop, dims, lower, upper):

	sample = np.zeros((dims+1, dims))

	sample[0][0:dims] = pop[-1][1:dims+1]

	for i, s in enumerate(np.random.randint(1, len(pop), dims)):
		sample[i+1][0:dims] = pop[s][1:dims+1]

	centroid = np.mean(sample, axis=0)
	proposal = np.subtract(centroid*2, sample[-1])
	if np.all(lower < proposal) and np.all(proposal < upper):
		return proposal
	else:
		try:
			return generate_proposal(pop, dims, lower, upper)
		except:
			return centroid # the error should never happen, but just in case :)


def generate_reflection(pop, proposal, dims, lower, upper):

	x0 = pop[-1][1:dims+1]

	reflection = np.zeros((dims))

	for i in range(0, dims):
		x = np.random.uniform()
		reflection[i] = (1+x)*x0[i] - x*proposal[i]

	if np.all(lower < reflection) and np.all(reflection < upper):
		return reflection
	else:
		try:
			return generate_reflection(pop, proposal, dims, lower, upper)
		except:
			return proposal # the above error should be very rare, but when it happens let's just return the proposal


def CRS2_LM(func, lower_bounds, upper_bounds, distribution_init, mean_init, sd_init, 
	init_pop_size, final_pop_size, max_iter, tol, processes):

	dims = len(lower_bounds)
	
	pop = np.zeros((init_pop_size, dims+1), float)

	crs_arg_list = []

	for i in range(0, init_pop_size):
			for j in range(1, dims+1):
				pop[i][j] = generate_point(distribution_init[j-1], mean_init[j-1], sd_init[j-1], 
					lower_bounds[j-1], upper_bounds[j-1])
			if processes <= 1:
				pop[i][0] = func(pop[i][1:dims+1])
			else:
				crs_arg_list.append(pop[i][1:dims+1])
	
	if processes > 1:
		with poolcontext(processes=processes) as pool:
			for i, result in enumerate(pool.imap(func, crs_arg_list)):
				pop[i][0] = result

		

	pop = pop[pop[:, 0].argsort()]
	#print(pop)

	iters_so_far = init_pop_size

	while iters_so_far < max_iter:

		pop = pop[pop[:, 0].argsort()] # sort pop by likelihood

		if pop[-1][0] - pop[0][0] < tol: # if tol reached, return best point
			return pop[-1]

		if iters_so_far % 2 == 0 and len(pop) > final_pop_size: # decrease pop size every two iterations
			pop = np.delete(pop, 0, 0)

		if processes <= 1:

			proposal = generate_proposal(pop, dims, lower_bounds, upper_bounds)
			loglikelihood = func(proposal)
			iters_so_far += 1
			if loglikelihood > pop[0][0]:
				pop[0][0] = loglikelihood
				pop[0][1:dims+1] = proposal
				
			else:
				reflection = generate_reflection(pop, proposal, dims, lower_bounds, upper_bounds)
				loglikelihood = func(reflection)
				iters_so_far += 1
				if loglikelihood > pop[0][0]:
					pop[0][0] = loglikelihood
					pop[0][1:dims+1] = reflection

		else:

			crs_arg_list = []
			crs_arg_list_refl = []
			for i in range(0, min(processes*4, final_pop_size//4)):
				crs_arg_list.append(generate_proposal(pop, dims, lower_bounds, upper_bounds))
			
			with poolcontext(processes=processes) as pool:
				for i, result in enumerate(pool.imap(func, crs_arg_list)):
					if result > pop[0][0]:
						pop[0][0] = result
						pop[0][1:dims+1] = crs_arg_list[i]
						pop = pop[pop[:, 0].argsort()]
					else:
						crs_arg_list_refl.append(generate_reflection(pop, crs_arg_list[i], dims, lower_bounds, upper_bounds))

			with poolcontext(processes=processes) as pool:
				for i, result in enumerate(pool.imap(func, crs_arg_list_refl)):
					if result > pop[0][0]:
						pop[0][0] = result
						pop[0][1:dims+1] = crs_arg_list_refl[i]
						pop = pop[pop[:, 0].argsort()]


			iters_so_far += len(crs_arg_list)
			iters_so_far += len(crs_arg_list_refl)

	return pop[-1]
