import itertools
from collections import defaultdict, deque
import random
import math
import pandas as pd
import statistics

import networkx as nx
from networkx.algorithms.community import modularity
from networkx.utils import py_random_state

import pickle
import sys
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

# Globals for paths
obj_path="../data/obj"
log_path="../logs/"
plot_path="../plots/"


# Calculates a weighted balance score for a partition of G
def weighted_intersectional_balance(G, partition, dems_dist, dems_weight):
	# Keep track of all demographic balance distributions
	balance_scores={}
	fexp_scores={}
	phi_scores={}
	nci_list=[]

	n=G.number_of_nodes()

	# Failsafe: If sum of dems_weight is 0, return.
	if sum([dems_weight[d] for d in dems_weight])==0: 
		print("Invalid weights for demographics.")
		return

	# For each demographic in the network:
	for d in dems_dist:
		# Get this demographic's attributes for the graph
		colors=nx.get_node_attributes(G,d)
		color_list=dems_dist[d].keys()
		K_cols=len(dems_dist[d])

		# Add dictionary entries to track individual community scores
		balance_scores[d]=[]
		fexp_scores[d]=[]

		# Calculate phi
		c_least=min([dems_dist[d][j] for j in color_list])
		phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)
		# Track phi score for demographic
		phi_scores[d]=phi

		# If n==0: return 0. Should not happen, but good failsafe
		if n==0: return 0

		sum_scores=0.0
		F_dist=[]

		# If color list length==1: balance is 1 by default
		if len(color_list)<=1:
			return [1.0 for _c in partition]

		# For all communities discovered
		for i, ci in enumerate(partition):
			# If community is populated:
			n_ci=len(ci)
			if n_ci>0:
				# First, calculate extra nodes for the community
				sum_dist=0
				for col in color_list:
					sum_dist+=math.floor(dems_dist[d][col]*n_ci/n)
				n_extra=n_ci-sum_dist

				# Calculate F_exp(c_i)
				f_exp=(K_cols*phi*n_ci-(phi+K_cols-1-(phi*K_cols))*n_extra) /((K_cols-1)*(K_cols*n_ci + (phi-1)*n_extra))

				## Also calculate F(c_i)
				# For all nodes u in ci, check sums of colors
				sum_cols=[0 for _c in color_list]
				for u in ci:
					# Extend for multiple colors
					for col_ind,col in enumerate(color_list):
						if colors[u]==col:
							sum_cols[col_ind]+=1
					
				min_balance=1.0
				# Iterate over all colors to find min balance for community
				for col_ind,col in enumerate(color_list):
					sum_color=sum_cols[col_ind]

					# If any sum==0, or the sum of the color==len(ci): Leave balance 0
					if sum_color==0 or sum_color==n_ci: 
						min_balance=0.0
						break

					# Otherwise: find if balance is min
					bal_score=sum_color/(n_ci-sum_color)
					if bal_score<min_balance:
						min_balance=bal_score

				# Set min_balance as the score. Normalize by K-cols st max score =1
				balance_ci=(K_cols-1)*min_balance

				# Get final score
				if n_ci>=K_cols:
					fscore_ci=min(1.0,1-(f_exp-balance_ci))
				else:
					fscore_ci=0.0

				# Append demographic balance and f_scores to lists.
				balance_scores[d].append(balance_ci)
				fexp_scores[d].append(balance_ci)

	
	balance_score_sum=0.0
	fexp_score_sum=0.0
	# Get final weighted average scores from dems_weight
	for d in dems_dist:
		bsum=0.0
		fsum=0.0
		# Weigh scores by community size
		for i,c in enumerate(partition):
			n_ci=len(c)
			bsum+=(n_ci*balance_scores[d][i])
			fsum+=(n_ci*fexp_scores[d][i])
		balance_score_sum+=(dems_weight[d]*bsum/n)
		fexp_score_sum+=(dems_weight[d]*fsum/n)
	# Get final scores
	overall_balance=balance_score_sum/sum([dems_weight[d] for d in dems_weight])
	overall_fexp=fexp_score_sum/sum([dems_weight[d] for d in dems_weight])

	# Return all tracked values
	return overall_balance, overall_fexp, balance_scores, fexp_scores, phi_scores

# @TODO: work on this
def full_intersectional_balance(G, partition, dems_dist):
	# for each combination of groups in all demographics, calculate balance
	return


def experiment(network,dems_weight,n_reps=3):
	# Load network
	with open(f"{obj_path}/{network}.nx","rb") as g_open:
		net=pickle.load(g_open)

	print(f"Network object {network} loaded.")
	print(f"{network}: N={net.number_of_nodes()}, M={net.number_of_edges()}")

	dems_dist={}

	# For each demographic, get value distribution
	for d in dems_weight:
		colors=nx.get_node_attributes(net,d)

		# Calculate demographic value ratios here
		dems_dist[d]={}

		# Calculate color distribution
		for n_ind in net.nodes():
			col=colors[n_ind]

			# Add to dict if does not exist
			if col not in dems_dist[d]:
				dems_dist[d][col]=1
			else:
				dems_dist[d][col]+=1

	print("Demographic distribution calculated.\n")
	for d in dems_dist:
		print(f"{d}:\n-----------")
		for c in dems_dist[d]:
			print(f"{c}:{dems_dist[d][c]}\n")


	# Run Louvain, get partition
	partition=nx.community.louvain_communities(net)

	# Get balance scores
	overall_balance,overall_fexp,balance_scores,fexp_scores,phi_scores = weighted_intersectional_balance(net, partition, dems_dist, dems_weight)


	print(f"Overall weighted balance: {overall_balance}")
	print(f"Overall weighted prop fairness: {overall_fexp}")

	print(f"Distribution of weighted balance: {balance_scores}")
	print(f"Distribution of weighted prop fairness: {fexp_scores}")

	print(f"Phi scores: {phi_scores}")


# Experiment in this setting using Louvain
if __name__ == '__main__':
	#experiment("proximity_dem",{"subject":1.0,"gender":1.0},2)
	experiment("facebook_107_dem",{"gender":1.0,"education":1.0},2)




