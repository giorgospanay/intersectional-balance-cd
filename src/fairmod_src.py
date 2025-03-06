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


### --------------- MOUFLON MAIN CALL ---------------------

@py_random_state("seed")
def fair_louvain_communities(
	G, weight="weight", resolution=1, threshold=0.0000001, max_level=None, seed=None, color_list=["blue","red"], color_dict={}, dem_weight={}, alpha=0.9, strategy="base"
):	
	partitions=[]

	# base strategy (fair-mod)
	if strategy=="base":
		partitions = fair_louvain_partitions_base(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)
	
	# step2 strategy: optimize for modularity first, then switch to fairness
	elif strategy=="step2":
		partitions = fair_louvain_partitions_step2(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)

	# fexp strategy: like base strategy, but also adds F_expected penalty
	elif strategy=="fexp":
		partitions = fair_louvain_partitions_fexp(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)

	# hybrid strategy: combines step2 and fexp penalty
	elif strategy=="hybrid":
		partitions = fair_louvain_partitions_hybrid(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)

	## @TODO: -- Multi-demographic strategies

	# base-comb: combination of demographics as unique colours.
	## Can process and run similarly to base
	elif strategy=="base_comb":
		partitions = fair_louvain_partitions_base_comb(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)

	# base-weight: each demographic receives its own weight
	elif strategy=="base_weight":
		partitions = fair_louvain_partitions_base_weight(G, color_dict, dem_weight, weight, resolution, threshold, seed, alpha=alpha)

	if max_level is not None:
		if max_level <= 0:
			raise ValueError("max_level argument must be a positive integer or None")
		partitions = itertools.islice(partitions, max_level)
	final_partition = deque(partitions, maxlen=1)
	return final_partition.pop()



### ------------ SELECT OPTIMIZATION STRATEGY ----------------


## Strategy "base"
## Base approach from Fair-mod paper: optimize the full Obj equation
@py_random_state("seed")
def fair_louvain_partitions_base(G, weight="weight", resolution=1, threshold=0.000001, seed=None, color_list=["blue","red"], alpha=0.9):
	partition = [{u} for u in G.nodes()]
	
	K_cols=len(color_list)
	# If one colour: stop. @TODO: revert to simple louvain?
	if K_cols==1:
		yield partition
		return

	colors=nx.get_node_attributes(G, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in G.nodes():
		color_dist[colors[n_ind]]+=1

	# Also calculate phi=overall balance of colours in G
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)


	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)
	
	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		graph.add_nodes_from(G)
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))

	# Set n, m
	n = graph.number_of_nodes()
	m = graph.size(weight="weight")


	# Prepare partition colours
	partition_colors_new=list()
	# For each community in partition:
	for comm in partition:
		n_ci=len(comm)
		if n_ci>0:
			# For all nodes u in ci, check sums of colors
			sum_cols=[0 for _c in color_list]
			for u in comm:
				# Extend for multiple colors
				for col_ind,col in enumerate(color_list):
					if colors[u]==col:
						sum_cols[col_ind]+=1
				
			min_balance=1.0
			# Iterate over all colors to find min balance for community
			for col_ind,col in enumerate(color_list):
				sum_color=sum_cols[col_ind]

				# If any sum==0, or the sum of the color==len(ci): Leave balance to 0
				if sum_color==0 or sum_color==n_ci: 
					min_balance=0.0
					break

				# Otherwise: find if balance is min
				bal_score=sum_color/(n_ci-sum_color)
				if bal_score<min_balance:
					min_balance=bal_score

			# Set min_balance as the score. Normalize by comm size, times K-1
			balance_ci=(K_cols-1)*min_balance*n_ci/n

			# Get final score
			fscore_ci=balance_ci

		# Generate partition dict
		p_dict={}
		for col_ind,col in enumerate(color_list):
			p_dict[col]=sum_cols[col_ind]
		p_dict["score"]=fscore_ci

		# Append to list
		partition_colors_new.append(p_dict)

	
	# Run using full Obj
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition_obj(
		graph, 
		n,
		m,
		partition,
		colors, 
		color_dist,
		partition_colors_new,
		phi,
		alpha=alpha,
		resolution=resolution,
		threshold=threshold, 
		is_directed=is_directed, 
		seed=seed
	)

	# Continue using full Obj=a*Q+(1-a)*F for improvements. Start with opt=0
	improvement = True
	opt=0
	while improvement:
		yield [s.copy() for s in partition]

		# Calculate new modularity, fairness and Obj scores
		new_mod = modularity(
			graph, inner_partition, resolution=resolution, weight="weight"
		)
		new_fair, _new_f_dist = fairness_base(
			G, partition, color_dist
		)
		new_opt = alpha * new_mod + (1-alpha) * new_fair

		# ...and stop optimizing if gain is less than threshold
		if new_opt - opt <= threshold:
			return

		mod = new_mod
		fair = new_fair
		opt = new_opt

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)

		# Refresh partition colors
		partition_colors = partition_colors_new

		# Run for improvement again using full Obj
		partition, inner_partition, improvement, partition_colors_new = _calculate_partition_obj(
			graph, 
			n,
			m,
			partition,
			colors, 
			color_dist,
			partition_colors_new,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)



## Strategy "step2"
## Calculate partitions, first running only for modularity, and then optimizing for Q,F
@py_random_state("seed")
def fair_louvain_partitions_step2(G, weight="weight", resolution=1, threshold=0.00001, seed=None, color_list=["blue","red"], alpha=0.9):
	partition = [{u} for u in G.nodes()]

	K_cols=len(color_list)
	# If one colour: stop. @TODO: revert to simple louvain?
	if K_cols==1:
		yield partition
		return

	colors=nx.get_node_attributes(G, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in G.nodes():
		color_dist[colors[n_ind]]+=1

	# Also calculate phi=overall balance of colours in G
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)


	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)
	
	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		graph.add_nodes_from(G)
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))

	n = graph.number_of_nodes()
	m = graph.size(weight="weight")

	# Prepare partition colours
	partition_colors_new=list()
	# For each community in partition:
	for comm in partition:
		n_ci=len(comm)
		if n_ci>0:
			# For all nodes u in ci, check sums of colors
			sum_cols=[0 for _c in color_list]
			for u in comm:
				# Extend for multiple colors
				for col_ind,col in enumerate(color_list):
					if colors[u]==col:
						sum_cols[col_ind]+=1
				
			min_balance=1.0
			# Iterate over all colors to find min balance for community
			for col_ind,col in enumerate(color_list):
				sum_color=sum_cols[col_ind]

				# If any sum==0, or the sum of the color==len(ci): Leave balance to 0
				if sum_color==0 or sum_color==n_ci: 
					min_balance=0.0
					break

				# Otherwise: find if balance is min
				bal_score=sum_color/(n_ci-sum_color)
				if bal_score<min_balance:
					min_balance=bal_score

			# Set min_balance as the score. Normalize by comm size, times K-1
			balance_ci=(K_cols-1)*min_balance*n_ci/n

			# Get final score
			fscore_ci=balance_ci

		# Generate partition dict
		p_dict={}
		for col_ind,col in enumerate(color_list):
			p_dict[col]=sum_cols[col_ind]
		p_dict["score"]=fscore_ci

		# Append to list
		partition_colors_new.append(p_dict)


	# First step calculates only modularity gain
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition_mod(
		graph, 
		n,
		m,
		partition, 
		colors,
		color_dist,
		phi,
		resolution=resolution, 
		is_directed=is_directed, 
		seed=seed
	)

	# Now start using full Obj=a*Q+(1-a)*F for improvements
	improvement = True
	first_fair_step = True
	while improvement:
		yield [s.copy() for s in partition]

		# For first step using fairness: calculate new round of improvement regardless
		if first_fair_step:
			# Set opt as previous (modularity only) optimum
			opt = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			# Remove flag
			first_fair_step=False

		# Otherwise check for improvement on Obj
		else:	
			new_mod = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			new_fair, _new_f_dist = fairness_base(
				G, partition, color_dist
			)

			new_opt = alpha * new_mod + (1-alpha) * new_fair

			if new_opt - opt <= threshold:
				return

			mod = new_mod
			fair = new_fair
			opt = new_opt

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)

		# Refresh partition colors
		partition_colors = partition_colors_new

		# Run for improvement again using full Obj
		partition, inner_partition, improvement, partition_colors_new = _calculate_partition_obj(
			graph, 
			n,
			m,
			partition,
			colors, 
			color_dist,
			partition_colors_new,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)



## Strategy "fexp"
## Approach adding penalty for F_expected
@py_random_state("seed")
def fair_louvain_partitions_fexp(G, weight="weight", resolution=1, threshold=0.000001, seed=None, color_list=["blue","red"], alpha=0.9):
	partition = [{u} for u in G.nodes()]

	K_cols=len(color_list)
	# If one colour: stop. @TODO: revert to simple louvain?
	if K_cols==1:
		yield partition
		return

	colors=nx.get_node_attributes(G, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in G.nodes():
		color_dist[colors[n_ind]]+=1

	# Also calculate phi=overall balance of colours in G
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)

	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)
	
	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		graph.add_nodes_from(G)
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))

	# Set n, m
	n = graph.number_of_nodes()
	m = graph.size(weight="weight")

	# Prepare partition colors
	# Now calculate all colours for generated partitioning
	partition_colors_new=list()
	# For each community in partition:
	for comm in partition:
		n_ci=len(comm)
		if n_ci>0:
			# First, calculate extra nodes for the community
			sum_dist=0
			for col in color_list:
				sum_dist+=math.floor(color_dist[col]*n_ci/n)
			n_extra=n_ci-sum_dist

			# Calculate F_exp(c_i)
			f_exp=(K_cols*phi*n_ci-(phi+K_cols-1-(phi*K_cols))*n_extra)/((K_cols-1)*(K_cols*n_ci + (phi-1)*n_extra))

			# For all nodes u in ci, check sums of colors
			sum_cols=[0 for _c in color_list]
			for u in comm:
				# Extend for multiple colors
				for col_ind,col in enumerate(color_list):
					if colors[u]==col:
						sum_cols[col_ind]+=1
				
			min_balance=1.0
			# Iterate over all colors to find min balance for community
			for col_ind,col in enumerate(color_list):
				sum_color=sum_cols[col_ind]

				# If any sum==0, or the sum of the color==len(ci): Leave balance to 0
				if sum_color==0 or sum_color==n_ci: 
					min_balance=0.0
					break

				# Otherwise: find if balance is min
				bal_score=sum_color/(n_ci-sum_color)
				if bal_score<min_balance:
					min_balance=bal_score

			# Set min_balance as the score. Normalize by comm size, times K-1
			balance_ci=(K_cols-1)*min_balance#/len(comm)

			# Get final score
			if n_ci>=K_cols:
				fscore_ci=min(1.0,1-(f_exp-balance_ci))
			else:
				fscore_ci=0.0

		# Generate partition dict
		p_dict={}
		for col_ind,col in enumerate(color_list):
			p_dict[col]=sum_cols[col_ind]
		p_dict["score"]=fscore_ci

		# Append to list
		partition_colors_new.append(p_dict)

	
	# Run using full Obj
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition_fexp(
		graph, 
		n,
		m,
		partition,
		colors, 
		color_dist,
		partition_colors_new,
		phi,
		alpha=alpha,
		resolution=resolution,
		threshold=threshold, 
		is_directed=is_directed, 
		seed=seed
	)

	# Continue using full Obj=a*Q+(1-a)*F for improvements. Start with opt=0
	improvement = True
	opt=0
	while improvement:
		yield [s.copy() for s in partition]

		# Calculate new modularity, fairness and Obj scores
		new_mod = modularity(
			graph, inner_partition, resolution=resolution, weight="weight"
		)
		new_fair, _new_f_dist = fairness_fexp(
			G, partition, color_dist
		)
		new_opt = alpha * new_mod + (1-alpha) * new_fair

		# # @DEBUG
		# print(f"------MAIN LOOP------")
		# print(f"opt={opt}, new_opt={new_opt} (new_mod={new_mod}, new_fair={new_fair})")

		# ...and stop optimizing if gain is less than threshold
		if new_opt - opt <= threshold:
			return

		mod = new_mod
		fair = new_fair
		opt = new_opt

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)

		# Refresh partition colors
		partition_colors = partition_colors_new

		# Run for improvement again using full Obj
		partition, inner_partition, improvement, partition_colors_new = _calculate_partition_fexp(
			graph, 
			n,
			m,
			partition,
			colors, 
			color_dist,
			partition_colors_new,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)


## Strategy "hybrid"
## Hybrid approach: Step2 + penalty for F_expected
@py_random_state("seed")
def fair_louvain_partitions_hybrid(G, weight="weight", resolution=1, threshold=0.000001, seed=None, color_list=["blue","red"], alpha=0.9):
	partition = [{u} for u in G.nodes()]

	K_cols=len(color_list)
	# If one colour: stop. @TODO: revert to simple louvain?
	if K_cols==1:
		yield partition
		return

	colors=nx.get_node_attributes(G, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in G.nodes():
		color_dist[colors[n_ind]]+=1

	# Also calculate phi=overall balance of colours in G
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)

	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)
	
	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		graph.add_nodes_from(G)
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))

	# Set n, m
	n = graph.number_of_nodes()
	m = graph.size(weight="weight")

	# Prepare partition colors
	# Now calculate all colours for generated partitioning
	partition_colors_new=list()
	# For each community in partition:
	for comm in partition:
		n_ci=len(comm)
		if n_ci>0:
			# First, calculate extra nodes for the community
			sum_dist=0
			for col in color_list:
				sum_dist+=math.floor(color_dist[col]*n_ci/n)
			n_extra=n_ci-sum_dist

			# Calculate F_exp(c_i)
			f_exp=(K_cols*phi*n_ci-(phi+K_cols-1-(phi*K_cols))*n_extra) /((K_cols-1)*(K_cols*n_ci + (phi-1)*n_extra))


			# For all nodes u in ci, check sums of colors
			sum_cols=[0 for _c in color_list]
			for u in comm:
				# Extend for multiple colors
				for col_ind,col in enumerate(color_list):
					if colors[u]==col:
						sum_cols[col_ind]+=1
				
			min_balance=1.0
			# Iterate over all colors to find min balance for community
			for col_ind,col in enumerate(color_list):
				sum_color=sum_cols[col_ind]

				# If any sum==0, or the sum of the color==len(ci): Leave balance to 0
				if sum_color==0 or sum_color==n_ci: 
					min_balance=0.0
					break

				# Otherwise: find if balance is min
				bal_score=sum_color/(n_ci-sum_color)
				if bal_score<min_balance:
					min_balance=bal_score

			# Set min_balance as the score. Normalize by comm size, times K-1
			balance_ci=(K_cols-1)*min_balance#/len(comm)

			# Get final score
			if n_ci>=K_cols:
				fscore_ci=min(1.0,1-(f_exp-balance_ci))
			else:
				fscore_ci=0.0

		# Generate partition dict
		p_dict={}
		for col_ind,col in enumerate(color_list):
			p_dict[col]=sum_cols[col_ind]
		p_dict["score"]=fscore_ci

		# Append to list
		partition_colors_new.append(p_dict)

	
	# First step calculates only modularity gain
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition_mod(
		graph,
		n, 
		m,
		partition, 
		colors,
		color_dist,
		phi,
		resolution=resolution, 
		is_directed=is_directed, 
		seed=seed,
		mode="fexp" # Set mode flag to calculate fairness score using F_exp
	)

	# Continue using full Obj=a*Q+(1-a)*F for improvements, with F_exp
	improvement = True
	first_fair_step = True
	opt=0
	while improvement:
		yield [s.copy() for s in partition]

		# # Calculate new modularity, fairness and Obj scores
		# new_mod = modularity(
		# 	graph, inner_partition, resolution=resolution, weight="weight"
		# )
		# new_fair, _new_f_dist = fairness_fexp(
		# 	G, partition, color_dist
		# )
		# new_opt = alpha * new_mod + (1-alpha) * new_fair


		# print(f"------MAIN LOOP------")
		# print(f"opt={opt}, new_opt={new_opt} (new_mod={new_mod}, new_fair={new_fair})")

		# For first step using fairness: calculate new round of improvement regardless
		if first_fair_step:
			# Set opt as previous (modularity only) optimum
			opt = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			# Remove flag
			first_fair_step=False

		# Otherwise check for improvement on Obj
		else:	
			new_mod = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			new_fair, _new_f_dist = fairness_fexp(
				G, partition, color_dist
			)

			new_opt = alpha * new_mod + (1-alpha) * new_fair

			if new_opt - opt <= threshold:
				return

			mod = new_mod
			fair = new_fair
			opt = new_opt


		# # ...and stop optimizing if gain is less than threshold
		# if new_opt - opt <= threshold:
		# 	return

		# mod = new_mod
		# fair = new_fair
		# opt = new_opt

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)

		# Refresh partition colors
		partition_colors = partition_colors_new

		# Then continue using full Obj (with F_expected penalty)
		partition, inner_partition, improvement, partition_colors_new = _calculate_partition_fexp(
			graph, 
			n,
			m,
			partition,
			colors, 
			color_dist,
			partition_colors_new,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)


##-----------------------------------------------------------------------------------

## Strategy "base_comb"
## Base approach from Fair-mod paper: optimize the full Obj equation.
## Instead, run balance for all demographics in the network 
@py_random_state("seed")
def fair_louvain_partitions_base_comb(G, weight="weight", resolution=1, threshold=0.000001, seed=None, color_list=["blue","red"], alpha=0.9):
	pass

	## Use for weight
	# # Calculate network color ratios here
	# color_dist={}
	# for d in color_dict:
	# 	color_dist[d]={}
	# 	for c in color_dict[d]:
	# 		color_dist[d][c]=0

	# for n_ind in G.nodes():
	# 	for d in color_dict:
	# 		color_dist[d][colors_dem[d][n_ind]]+=1


## Strategy "base_weight"
## Base approach from Fair-mod paper: optimize the full Obj equation.
## Multiple demographics strategy: provide weight matrix, fairness is weighted avg
@py_random_state("seed")
def fair_louvain_partitions_base_weight(G, color_dict, dem_weight, weight="weight", resolution=1, threshold=0.000001, seed=None, alpha=0.9):
	partition = [{u} for u in G.nodes()]
	
	# Calculate K for each demographic
	K_cols={d:len(color_dict[d]) for d in color_dict}

	# Get colors for each demographic
	colors_dem={d:nx.get_node_attributes(G,d) for d in color_dict}

	# Calculate network demographic ratios here
	color_dist={}
	for d in color_dict:
		color_dist[d]={}
		for c in color_dict[d]:
			color_dist[d][c]=0

	for n_ind in G.nodes():
		for d in color_dict:
			color_dist[d][colors_dem[d][n_ind]]+=1


	# If any of the demographics has one colour: stop. @TODO: revert to simple louvain?
	for d in color_dict:
		if K_cols[d]==1:
			yield partition
			return

	# Also calculate phi=overall weighted balance of colours in G
	c_least_dem={d:min([color_dist[d][c] for c in color_dist]) for d in color_dist}
	phi_dem={d:(K_cols[d]-1)*c_least_dem[d]/(len(G.nodes())-c_least_dem[d]) for d in color_dist}

	phi=sum([dem_weight[d]*phi_dem[d] for d in color_dist])/sum([dem_weight[d] for d in color_dist])

	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)
	
	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		graph.add_nodes_from(G)
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))

	# Set n, m
	n = graph.number_of_nodes()
	m = graph.size(weight="weight")


	# Prepare partition colours
	partition_colors_new=list()
	# For each community in partition:
	for comm in partition:
		n_ci=len(comm)
		if n_ci>0:
			# For all nodes u in ci, check sums of colors
			sum_cols={d:[0 for _c in color_dist[d]] for d in color_dist} 
			for u in comm:
				# Extend for multiple colors
				for d in color_dist:
					for col_ind,col in enumerate(color_dist[d]):
						if colors_dem[d][u]==col:
							sum_cols[d][col_ind]+=1
				
			min_balance=1.0
			balance_dem={}
			fscore_dem={}
			# Iterate over all demographics and colors to find min balance for community
			for d in color_dist:
				for col_ind,col in enumerate(color_dist[d]):
					sum_color=sum_cols[d][col_ind]

					# If any sum==0, or the sum of the color==len(ci): Leave balance to 0
					if sum_color==0 or sum_color==n_ci: 
						min_balance=0.0
						break

					# Otherwise: find if balance is min
					bal_score=sum_color/(n_ci-sum_color)
					if bal_score<min_balance:
						min_balance=bal_score

				# Set min_balance as the score. Normalize by comm size, times K-1
				balance_dem[d]=(K_cols-1)*min_balance*n_ci/n

				# Get final score
				fscore_dem[d]=balance_ci[d]

		# Generate partition dict
		p_dict={}
		for d in color_dist:
			for col_ind,col in enumerate(color_dist[d]):
				p_dict[col]=sum_cols[d][col_ind]
			# Set score for each demographic
			p_dict[f"{d}_score"]=fscore_dem[d]
		# and after weighting all scores
		p_dict["score"]=sum([dem_weight[d]*p_dict[f"{d}_score"] for d in color_dist])/sum([dem_weight[d] for d in color_dist])

		# Append to list
		partition_colors_new.append(p_dict)

	# Run using full Obj
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition_obj_weighted(
		graph, 
		n,
		m,
		partition,
		colors, 
		color_dist,
		partition_colors_new,
		phi,
		alpha=alpha,
		resolution=resolution,
		threshold=threshold, 
		is_directed=is_directed, 
		seed=seed
	)

	# Continue using full Obj=a*Q+(1-a)*F for improvements. Start with opt=0
	improvement = True
	opt=0
	while improvement:
		yield [s.copy() for s in partition]

		# Calculate new modularity, fairness and Obj scores
		new_mod = modularity(
			graph, inner_partition, resolution=resolution, weight="weight"
		)
		new_fair, _new_f_dist = fairness_base(
			G, partition, color_dist
		)
		new_opt = alpha * new_mod + (1-alpha) * new_fair

		# ...and stop optimizing if gain is less than threshold
		if new_opt - opt <= threshold:
			return

		mod = new_mod
		fair = new_fair
		opt = new_opt

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)

		# Refresh partition colors
		partition_colors = partition_colors_new

		# Run for improvement again using full Obj
		partition, inner_partition, improvement, partition_colors_new = _calculate_partition_obj_weighted(
			graph, 
			n,
			m,
			partition,
			colors, 
			color_dist,
			partition_colors_new,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)







# ------------- CALCULATE SINGLE LEVEL PARTITIONS --------------------

# Calculate one level of partitions based on modularity alone
@py_random_state("seed")
def _calculate_partition_mod(G, n, m, partition, colors, color_dist, phi, resolution=1, is_directed=False, seed=None, mode="base"):
	# At start, assign each node to a community 
	comms = {u: i for i,u in enumerate(G.nodes())}
	inner_partition = [{u} for u in G.nodes()]

	color_list=color_dist.keys()
	K_cols=len(color_list)

	original_partition=partition.copy()

	# @TODO: fix when supporting DiGraphs
	if is_directed==True:
		print("Directed networks not supported.")
		return None

	# Get all neighbours, including R/B
	nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
	
	# Get sum of edge weights for all B/R nodes
	sum_all = {u: len(nbrs[u]) for u in G}

	degrees=dict(G.degree(weight="weight"))
	Stot=list(degrees.values())

	# Do random shuffle on seed
	rand_nodes=list(G.nodes)
	seed.shuffle(rand_nodes)


	# Start moving nodes here
	n_moves=1
	improvement=False

	while n_moves>0:
		n_moves=0
		for u in rand_nodes:
			mod_best=0
			comm_best=comms[u]

			# Get degrees
			degree=degrees[u]
			Stot[comm_best]-=degree

			# For each node, calculate the weights of its neighbours in the same comm
			w2c=neighbor_weights(nbrs[u],comms)

			# Calculate modularity remove cost
			mod_rem = -w2c[comm_best]/m + resolution*(Stot[comm_best]*degree)/(2* m**2)

			# For each neighbour of u: 
			for nbr_comm, wt in w2c.items():
				# Calculate modularity gain
				mod_gain = mod_rem + wt/m - resolution*(Stot[nbr_comm]*degree)/(2* m**2)
				# If improves over previous best, assign as best.
				if mod_gain > mod_best:
					# # @DEBUG
					# print(f"Move {u}->{nbr_comm} found:")
					# print(f"mod_best={mod_best}, mod_gain={mod_gain}")

					mod_best = mod_gain
					comm_best = nbr_comm

			# Then finalize move if necessary
			Stot[comm_best]+=degree
			if comm_best != comms[u]:
				# Finalize move

				# Get nodes in com
				com = G.nodes[u].get("nodes", {u})

				# Update partition, remove com nodes from comms(u)
				partition[comms[u]].difference_update(com)
				# Update inner partition, remove u from comms(u)
				inner_partition[comms[u]].remove(u)
				# Update partition, add com nodes to comm_best
				partition[comm_best].update(com)
				# Update inner partition, add u to comm_best
				inner_partition[comm_best].add(u)
				# Signal improvement
				improvement = True
					
				# Comment for infinite loop on a=0
				n_moves += 1

				# Change new best community for u
				comms[u] = comm_best

	# Filter out all empty partitions
	partition = list(filter(len, partition))
	inner_partition = list(filter(len, inner_partition))


	# Now calculate all colours for generated partitioning
	partition_colors_new=list()
	# For each community in partition:
	for comm in partition:
		n_ci=len(comm)
		if n_ci>0:
			# For all nodes u in ci, check sums of colors
			sum_cols=[0 for _c in color_list]
			for u in comm:
				# Extend for multiple colors
				for col_ind,col in enumerate(color_list):
					if colors[u]==col:
						sum_cols[col_ind]+=1
				
			min_balance=1.0
			# Iterate over all colors to find min balance for community
			for col_ind,col in enumerate(color_list):
				sum_color=sum_cols[col_ind]

				# If any sum==0, or the sum of the color==len(ci): Leave balance to 0
				if sum_color==0 or sum_color==n_ci: 
					min_balance=0.0
					break

				# Otherwise: find if balance is min
				bal_score=sum_color/(n_ci-sum_color)
				if bal_score<min_balance:
					min_balance=bal_score

			# Set min_balance as the score. Normalize by comm size
			balance_ci=(K_cols-1)*min_balance

			# Originally set fscore as balance_ci
			fscore_ci=balance_ci

			# Change score to fexp calculation if mode is fexp
			if mode=="fexp":
				# First, calculate extra nodes for the community
				sum_dist=0
				for col in color_list:
					sum_dist+=math.floor(color_dist[col]*n_ci/n)
				n_extra=n_ci-sum_dist

				# Calculate F_exp(c_i)
				f_exp=(K_cols*phi*n_ci-(phi+K_cols-1-(phi*K_cols))*n_extra) /((K_cols-1)*(K_cols*n_ci + (phi-1)*n_extra))

				# Get final score
				if n_ci>=K_cols:
					fscore_ci=min(1.0,1.0-(f_exp-balance_ci))
				else:
					fscore_ci=0.0

		# Generate partition dict
		p_dict={}
		for col_ind,col in enumerate(color_list):
			p_dict[col]=sum_cols[col_ind]
		p_dict["score"]=fscore_ci

		# Append to list
		partition_colors_new.append(p_dict)

	return partition, inner_partition, improvement, partition_colors_new



# Calculate one level of partitions on full Obj=a*Q+(1-a)*F
@py_random_state("seed")
def _calculate_partition_obj(G, n, m, partition, colors, color_dist, partition_cols, phi, alpha=0.9, resolution=1, threshold=0.0000001, is_directed=False, seed=None):
	# At start, assign each node to a community 
	comms = {u: i for i,u in enumerate(G.nodes())}
	inner_partition = [{u} for u in G.nodes()]

	color_list=color_dist.keys()
	K_cols=len(color_list)

	partition_colors=partition_cols.copy()
	original_partition=partition.copy()

	# @TODO: fix when supporting DiGraphs
	if is_directed==True:
		print("Directed networks not supported.")
		return None

	# Get all neighbours, including R/B
	nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
	
	# Get sum of edge weights for all B/R nodes
	sum_all = {u: len(nbrs[u]) for u in G}

	degrees=dict(G.degree(weight="weight"))
	Stot=list(degrees.values())


	# Do random shuffle on seed
	rand_nodes=list(G.nodes)
	seed.shuffle(rand_nodes)


	# Start calculating movements to other communities
	n_moves=1
	improvement=False

	while n_moves>0:
		n_moves=0
			
		# For each node:
		for u in rand_nodes:
			## Start with opt score for its current community?
			#  Idea: the minimum gain score is 0 for modularity and the previous fair
			#     score for the current community =>a*0+(1-a)*curr score
			opt_best=0.0
			#opt_best=(1-alpha)*partition_colors[comms[u]]["score"]
			comm_best=comms[u]
			new_fair_update=0.0
			post_fair_update=0.0

			# Get degrees
			degree=degrees[u]
			Stot[comm_best]-=degree

			# For each node, calculate the weights of its neighbours in the same comm
			w2c=neighbor_weights(nbrs[u],comms)

			
			# Calculate modularity remove cost
			mod_rem = -w2c[comm_best]/m + resolution*(Stot[comm_best]*degree)/(2* m**2)

			# Calculate fairness (balance) current cost
			#### @TODO: slowest part, probably. Can be optimized?
			com = G.nodes[u].get("nodes", {u})


			# For nodes under u: calculate their color scores
			curr_colors=[0 for _col in color_list]
			for i in com:
				for col_ind, col in enumerate(color_list):
					if colors[i]==col:
						curr_colors[col_ind]+=1

			# Curr fair: all nodes in comms(u)
			all_colors=[partition_colors[comms[u]][col] for col in color_list]			
	
			min_fair=1.0
			sum_pnodes=0
			for col_ind, col in enumerate(color_list):
				sum_color=all_colors[col_ind]
				sum_pnodes+=sum_color
				# Min=0 if any color is 0
				if sum_color==0: 
					min_fair=0.0
				else:
					up_score=0.0
					if len(com)>sum_color:
						# See if minimum color score is less than current min
						up_score=sum_color/(len(com)-sum_color)

					# Set to min if less than current
					if up_score<min_fair:
						min_fair=up_score

			# normalize by community size ratio
			curr_fair=(K_cols-1)*min_fair*len(com)/n

			# Calculate post_fair score of current community after moving u
			post_colors=[all_colors[col_ind]-curr_colors[col_ind] for col_ind,_col in enumerate(color_list)]
			
			# Important: add failsafe here. If post drops <0, do not allow move. cont
			drop_below_flag=False
			for col_ind,_col in enumerate(color_list):
				if post_colors[col_ind]<0: 
					drop_below_flag=True
					break
			if drop_below_flag:
				continue

			# Now calculate post_fair
			post_min_fair=1.0
			post_comm_len=sum(post_colors)
			for col_ind,_col in enumerate(color_list):
				if post_colors[col_ind]==0:
					post_min_fair=0.0 
				else:
					post_col_score=0.0
					if post_comm_len>post_colors[col_ind]:
						post_col_score=post_colors[col_ind]/(post_comm_len-post_colors[col_ind])
					# Set to min if less than current
					if post_col_score<post_min_fair:
						post_min_fair=post_col_score
			# Weigh by community size ratio for post_fair
			post_fair=(K_cols-1)*post_comm_len*post_min_fair/n


			#print(f"Moving node {u}: curr_fair={curr_fair}, post_fair={post_fair}")


			# For each neighbour of u: 
			for nbr_comm, wt in w2c.items():

				# Calculate new modularity
				mod_gain = mod_rem + wt/m - resolution*(Stot[nbr_comm]*degree)/(2* m**2)

				# Calculate gain for move to nbr_comm
				new_curr_len=sum([partition_colors[nbr_comm][col] for col_ind,col in enumerate(color_list)])
				# Weigh by community size ratio. Assumes already normalized score
				new_score = partition_colors[nbr_comm]["score"]

				new_colors = [partition_colors[nbr_comm][col]+curr_colors[col_ind] for col_ind,col in enumerate(color_list)]
				# Calculate fairness score for newly created community (plus u)
				new_min_fair=1.0
				new_comm_len=sum(new_colors)
				for col_ind,_col in enumerate(color_list):
					if new_colors[col_ind]==0:
						new_min_fair=0.0 
					else:
						new_col_score=0.0
						if new_comm_len>new_colors[col_ind]:
							new_col_score=new_colors[col_ind]/(new_comm_len-new_colors[col_ind])
						# Set to min if less than current
						if new_col_score<new_min_fair:
							new_min_fair=new_col_score
				# Weigh by community length ratio for new_fair. Times Kcols
				new_fair=(K_cols-1) * new_min_fair * new_comm_len / n

				# Overall fairness score gain
				# new_fair (nbr) + post_fair (comms(u)) - new_score (nbr) - curr_fair (comms(u))
				fair_gain = (new_fair - new_score) + (post_fair - curr_fair)

				# Calculate new opt score.
				opt_gain = alpha * mod_gain + (1-alpha) * fair_gain

				# If opt gain tops previous gain found, set as new best
				if opt_gain > opt_best and (opt_gain - opt_best) > threshold:
					# # @DEBUG
					# print(f"Better found move: {u}->comm#{nbr_comm}")
					# print(f"u curr_f={curr_fair}, u post_f={post_fair}")
					# print(f"v curr_f={new_score}, v post_f={new_fair}")
					# print(f"fair gain={fair_gain}, mod_gain={mod_gain}")
					# print(f"opt gain={opt_gain}, (opt_best={opt_best})")
					# print("-----------")

					opt_best=opt_gain
					comm_best=nbr_comm

					new_fair_update=new_fair
					post_fair_update=post_fair


			# Then finalize move if necessary
			Stot[comm_best]+=degree
			if comm_best != comms[u]:
				# Finalize move

				# Get nodes in com
				com = G.nodes[u].get("nodes", {u})


				# Update partition, remove com nodes from comms(u)
				partition[comms[u]].difference_update(com)
				# Update inner partition, remove u from comms(u)
				inner_partition[comms[u]].remove(u)
				# Update partition, add com nodes to comm_best
				partition[comm_best].update(com)
				# Update inner partition, add u to comm_best
				inner_partition[comm_best].add(u)
				# Signal improvement
				improvement = True
					
				# Comment for infinite loop on a=0
				#n_moves += 1


				# Update new community colors
				for col_ind,col in enumerate(color_list):
					partition_colors[comm_best][col]+=curr_colors[col_ind]
				partition_colors[comm_best]["score"]=new_fair_update
				
				# Update old community colors
				for col_ind,col in enumerate(color_list):
					partition_colors[comms[u]][col]-=curr_colors[col_ind]
				partition_colors[comms[u]]["score"]=post_fair_update


				# Change new best community for u
				comms[u] = comm_best



	partition = list(filter(len, partition))
	inner_partition = list(filter(len, inner_partition))	
	partition_colors_new = list(filter(lambda ls: _full_partition_colors(ls,color_list), partition_colors))


	return partition, inner_partition, improvement, partition_colors_new



# Calculate one level of partitions based on full Obj, with F_exp penalty
@py_random_state("seed")
def _calculate_partition_fexp(G, n, m, partition, colors, color_dist, partition_cols, phi, alpha=0.9, resolution=1, threshold=0.0000001, is_directed=False, seed=None):
	# At start, assign each node to a community 
	comms = {u: i for i,u in enumerate(G.nodes())}
	inner_partition = [{u} for u in G.nodes()]

	color_list=color_dist.keys()
	K_cols=len(color_list)

	partition_colors=partition_cols.copy()
	original_partition=partition.copy()

	# @TODO: fix when supporting DiGraphs
	if is_directed==True:
		print("Directed networks not supported.")
		return None

	# Get all neighbours, including R/B
	nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
	
	# Get sum of edge weights for all B/R nodes
	sum_all = {u: len(nbrs[u]) for u in G}

	degrees=dict(G.degree(weight="weight"))
	Stot=list(degrees.values())

	# Do random shuffle on seed
	rand_nodes=list(G.nodes)
	seed.shuffle(rand_nodes)

	# Start calculating movements to other communities
	n_moves=1
	improvement=False


	while n_moves>0:
		n_moves=0
			
		# For each node:
		for u in rand_nodes:
			
			## Start with opt score for its current community?
			#  Idea: the minimum gain score is 0 for modularity and the previous fair
			#     score for the current community =>a*0+(1-a)*curr score
			opt_best=0.0
			#opt_best=(1-alpha)*partition_colors[comms[u]]["score"]
			comm_best=comms[u]
			new_fair_update=0.0
			post_fair_update=0.0

			# Get degrees
			degree=degrees[u]
			Stot[comm_best]-=degree

			# For each node, calculate the weights of its neighbours in the same comm
			w2c=neighbor_weights(nbrs[u],comms)

			
			# Calculate modularity remove cost
			mod_rem = -w2c[comm_best]/m + resolution*(Stot[comm_best]*degree)/(2* m**2)

			# Calculate fairness (balance) current cost
			#### @TODO: slowest part, probably. Can be optimized?
			com = G.nodes[u].get("nodes", {u})


			# For nodes under u: calculate their color scores
			curr_colors=[0 for _col in color_list]
			for i in com:
				for col_ind, col in enumerate(color_list):
					if colors[i]==col:
						curr_colors[col_ind]+=1

			# Curr fair: all nodes in comms(u)
			all_colors=[partition_colors[comms[u]][col] for col in color_list]			
	
			min_fair=1.0
			sum_pnodes=0
			for col_ind, col in enumerate(color_list):
				sum_color=all_colors[col_ind]
				sum_pnodes+=sum_color
				# Min=0 if any color is 0
				if sum_color==0: 
					min_fair=0.0
				else:
					up_score=0.0
					if len(com)>sum_color:
						# See if minimum color score is less than current min
						up_score=sum_color/(len(com)-sum_color)

					# Set to min if less than current
					if up_score<min_fair:
						min_fair=up_score

			# Current fairness score for u
			curr_fair=(K_cols-1)*min_fair


			# For current partition: calculate current F_exp
			# First, calculate extra nodes for the community
			sum_dist=0
			for col in color_list:
				sum_dist+=math.floor(color_dist[col]*len(com)/n)
			curr_n_extra=len(com)-sum_dist

			# Calculate current F_exp(c_i)
			curr_f_exp=(K_cols*phi*len(com)-(phi+K_cols-1-(phi*K_cols))*curr_n_extra) /((K_cols-1)*(K_cols*len(com) + (phi-1)*curr_n_extra))


			# Calculate score for current c_i minus the penalty
			if len(com)>=K_cols:
				curr_fscore=min(1.0,1-(curr_f_exp-((K_cols-1)*curr_fair)))
			else:
				curr_fscore=0.0



			# Calculate post_fair score of current community after moving u
			post_colors=[all_colors[col_ind]-curr_colors[col_ind] for col_ind,_col in enumerate(color_list)]
			
			# Important: add failsafe here. If post drops <0, do not allow move. cont
			drop_below_flag=False
			for col_ind,_col in enumerate(color_list):
				if post_colors[col_ind]<0: 
					drop_below_flag=True
					break
			if drop_below_flag:
				continue

			# Now calculate post_fair
			post_min_fair=1.0
			post_comm_len=sum(post_colors)
			for col_ind,_col in enumerate(color_list):
				if post_colors[col_ind]==0:
					post_min_fair=0.0 
				else:
					post_col_score=0.0
					if post_comm_len>post_colors[col_ind]:
						post_col_score=post_colors[col_ind]/(post_comm_len-post_colors[col_ind])
					# Set to min if less than current
					if post_col_score<post_min_fair:
						post_min_fair=post_col_score

			# Weigh by community length for post_fair
			post_fair=(K_cols-1)*post_min_fair


			# For post partition: calculate F_exp
			# First, calculate extra nodes for the community
			sum_dist=0
			for col in color_list:
				sum_dist+=math.floor(color_dist[col]*post_comm_len/n)
			post_n_extra=post_comm_len-sum_dist

			# Calculate current F_exp(c_i)
			if post_comm_len==0:
				post_f_exp=0.0
			else:
				post_f_exp=(K_cols*phi*post_comm_len-(phi+K_cols-1-(phi*K_cols))*post_n_extra) /((K_cols-1)*(K_cols*post_comm_len + (phi-1)*post_n_extra))

			# Calculate score for current c_i minus the penalty
			if post_comm_len>=K_cols:
				post_fscore=min(1.0,1-(post_f_exp-((K_cols-1)*post_fair)))
			else:
				post_fscore=0.0


			# Calculate current community loss from move
			loss_fscore=((post_fscore*post_comm_len)-(curr_fscore*len(com)))/n

			# # @DEBUG
			# print(f"Moving node #{u}. Curr_fair={curr_fair}, fscore={curr_fscore}, post_fair={post_fair}, post_fscore={post_fscore}, loss={loss_fscore}")


			# For each neighbour of u: 
			for nbr_comm, wt in w2c.items():

				# Calculate new modularity
				mod_gain = mod_rem + wt/m - resolution*(Stot[nbr_comm]*degree)/(2* m**2)

				# Calculate gain for move to nbr_comm
				new_curr_len=sum([partition_colors[nbr_comm][col] for col_ind,col in enumerate(color_list)])
				new_score = partition_colors[nbr_comm]["score"]
				## Assuming here: old score already has penalty on top


				new_colors = [partition_colors[nbr_comm][col]+curr_colors[col_ind] for col_ind,col in enumerate(color_list)]
				# Calculate fairness score for newly created community (plus u)
				new_min_fair=1.0
				new_comm_len=sum(new_colors)
				for col_ind,_col in enumerate(color_list):
					if new_colors[col_ind]==0:
						new_min_fair=0.0 
					else:
						new_col_score=0.0
						if new_comm_len>new_colors[col_ind]:
							new_col_score=new_colors[col_ind]/(new_comm_len-new_colors[col_ind])
						# Set to min if less than current
						if new_col_score<new_min_fair:
							new_min_fair=new_col_score
				new_fair=new_min_fair


				# Also calculate new score of v with penalty
				# For post partition: calculate F_exp
				# First, calculate extra nodes for the community
				sum_dist=0
				for col in color_list:
					sum_dist+=math.floor(color_dist[col]*new_comm_len/n)
				new_n_extra=new_comm_len-sum_dist

				# Calculate new F_exp(c_i)
				if new_comm_len==0:
					new_f_exp=0.0
				else:
					new_f_exp=(K_cols*phi*new_comm_len-(phi+K_cols-1-(phi*K_cols))*new_n_extra) /((K_cols-1)*(K_cols*new_comm_len + (phi-1)*new_n_extra))

				# Calculate score for current c_i minus the penalty
				if new_comm_len>=K_cols:
					new_fscore=min(1.0,1-(new_f_exp-((K_cols-1)*new_fair)))
				else:
					new_fscore=0.0

				# Calculate gain on v:
				gain_fscore=((new_fscore*new_comm_len)-(new_score*new_curr_len))/n


				
				### ------ UPDATE GAINS HERE ------
				# New gain including penalties: gain_v + loss_u
				# Overall fairness score gain with penalties (normalize weighted by n)
				fair_gain = (gain_fscore + loss_fscore)

				# Calculate new opt score
				opt_gain = alpha * mod_gain + (1-alpha) * fair_gain


				# # @DEBUG
				# print(f"Test ({u}->{nbr_comm}) opt_gain={opt_gain}, mod_gain={mod_gain}, fair_gain={fair_gain}")

				# If opt gain tops previous gain found, set as new best
				if opt_gain > opt_best and (opt_gain - opt_best) > threshold:


					# # @DEBUG
					# print(f"Better found move: {u}->comm#{nbr_comm}")
					# print(f"u curr_f={curr_fscore}, u post_f={post_fscore}")
					# print(f"v curr_f={new_score}, v post_f={new_fscore}")
					# print(f"fair gain={fair_gain}, mod_gain={mod_gain}")
					# print(f"opt gain={opt_gain}, (opt_best={opt_best})")
					# print("-----------")

					opt_best=opt_gain
					comm_best=nbr_comm

					new_fair_update=new_fscore
					post_fair_update=post_fscore


			# Then finalize move if necessary
			Stot[comm_best]+=degree
			if comm_best != comms[u]:
				# Finalize move

				# Get nodes in com
				com = G.nodes[u].get("nodes", {u})

				# Update partition, remove com nodes from comms(u)
				partition[comms[u]].difference_update(com)
				# Update inner partition, remove u from comms(u)
				inner_partition[comms[u]].remove(u)
				# Update partition, add com nodes to comm_best
				partition[comm_best].update(com)
				# Update inner partition, add u to comm_best
				inner_partition[comm_best].add(u)
				# Signal improvement
				improvement = True
					
				# Comment for infinite loop on a=0
				#n_moves += 1


				# Update new community colors
				for col_ind,col in enumerate(color_list):
					partition_colors[comm_best][col]+=curr_colors[col_ind]
				partition_colors[comm_best]["score"]=new_fair_update
				
				# Update old community colors
				for col_ind,col in enumerate(color_list):
					partition_colors[comms[u]][col]-=curr_colors[col_ind]
				partition_colors[comms[u]]["score"]=post_fair_update

				# Change new best community for u
				comms[u] = comm_best


	partition = list(filter(len, partition))
	inner_partition = list(filter(len, inner_partition))
	partition_colors_new = list(filter(lambda ls: _full_partition_colors(ls,color_list), partition_colors))
		

	return partition, inner_partition, improvement, partition_colors_new


# -----------------------------------------------------
# Calculate one level of partitions on full Obj=a*Q+(1-a)*F
#@TODO
@py_random_state("seed")
def _calculate_partition_obj_weighted(G, n, m, partition, colors, color_dist, partition_cols, phi, alpha=0.9, resolution=1, threshold=0.0000001, is_directed=False, seed=None):
	# At start, assign each node to a community 
	comms = {u: i for i,u in enumerate(G.nodes())}
	inner_partition = [{u} for u in G.nodes()]

	K_cols={d:len(color_dist[d]) for d in color_dist}

	partition_colors=partition_cols.copy()
	original_partition=partition.copy()

	# @TODO: fix when supporting DiGraphs
	if is_directed==True:
		print("Directed networks not supported.")
		return None

	# Get all neighbours, including R/B
	nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
	
	# Get sum of edge weights for all B/R nodes
	sum_all = {u: len(nbrs[u]) for u in G}

	degrees=dict(G.degree(weight="weight"))
	Stot=list(degrees.values())


	# Do random shuffle on seed
	rand_nodes=list(G.nodes)
	seed.shuffle(rand_nodes)


	# Start calculating movements to other communities
	n_moves=1
	improvement=False

	while n_moves>0:
		n_moves=0
			
		# For each node:
		for u in rand_nodes:
			## Start with opt score for its current community?
			#  Idea: the minimum gain score is 0 for modularity and the previous fair
			#     score for the current community =>a*0+(1-a)*curr score
			opt_best=0.0
			#opt_best=(1-alpha)*partition_colors[comms[u]]["score"]
			comm_best=comms[u]
			new_fair_update=0.0
			post_fair_update=0.0

			# Get degrees
			degree=degrees[u]
			Stot[comm_best]-=degree

			# For each node, calculate the weights of its neighbours in the same comm
			w2c=neighbor_weights(nbrs[u],comms)

			
			# Calculate modularity remove cost
			mod_rem = -w2c[comm_best]/m + resolution*(Stot[comm_best]*degree)/(2* m**2)

			# Calculate fairness (balance) current cost
			#### @TODO: slowest part, probably. Can be optimized?
			com = G.nodes[u].get("nodes", {u})

			# For nodes under u: calculate their color scores
			curr_colors={}
			for d in color_dist:
				curr_colors[d]=[0 for _col in color_dist[d]]
				for i in com:
					for col_ind, col in enumerate(color_dist[d]):
						if colors[d][i]==col:
							curr_colors[d][col_ind]+=1

			
			drop_below_flag=False
			all_colors={}
			post_colors={}
			curr_fair={}
			post_fair={}
			# For all demographics: calculate change of colors and fairness in comm(u)
			for d in color_dist:
				if drop_below_flag: continue

				all_colors[d]=[partition_colors[comms[u]][col] for col in color_dist[d]]			
				
				# Calculate curr_fair score of community
				min_fair=1.0
				sum_pnodes=0
				for col_ind, col in enumerate(color_dist[d]):
					sum_color=all_colors[d][col_ind]
					sum_pnodes+=sum_color
					# Min=0 if any color is 0
					if sum_color==0: 
						min_fair=0.0
					else:
						up_score=0.0
						if len(com)>sum_color:
							# See if minimum color score is less than current min
							up_score=sum_color/(len(com)-sum_color)

						# Set to min if less than current
						if up_score<min_fair:
							min_fair=up_score

				# normalize by community size ratio
				curr_fair[d]=(K_cols[d]-1)*min_fair*len(com)/n


				# Calculate post_fair score of current community after moving u
				post_colors[d]=[all_colors[d][col_ind]-curr_colors[d][col_ind] for col_ind,_col in enumerate(color_dist[d])]
				
				# Important: add failsafe here. If post drops <0, do not allow move. cont
				drop_below_flag=False
				for col_ind,_col in enumerate(color_dist[d]):
					if post_colors[d][col_ind]<0: 
						drop_below_flag=True
						break

				# Now calculate post_fair
				post_min_fair=1.0
				post_comm_len=sum(post_colors[d])
				for col_ind,_col in enumerate(color_dist[d]):
					if post_colors[d][col_ind]==0:
						post_min_fair=0.0 
					else:
						post_col_score=0.0
						if post_comm_len>post_colors[d][col_ind]:
							post_col_score=post_colors[d][col_ind]/(post_comm_len-post_colors[d][col_ind])
						# Set to min if less than current
						if post_col_score<post_min_fair:
							post_min_fair=post_col_score
				# Weigh by community size ratio for post_fair
				post_fair[d]=(K_cols[d]-1)*post_comm_len*post_min_fair/n


			# Calculate overall score
			

			# @DEBUG
			#print(f"Moving node {u}: curr_fair={curr_fair}, post_fair={post_fair}")


			# For each neighbour of u: 
			for nbr_comm, wt in w2c.items():

				# Calculate new modularity
				mod_gain = mod_rem + wt/m - resolution*(Stot[nbr_comm]*degree)/(2* m**2)

				# Calculate gain for move to nbr_comm
				new_curr_len=sum([partition_colors[nbr_comm][col] for col_ind,col in enumerate(color_list)])
				# Weigh by community size ratio. Assumes already normalized score
				new_score = partition_colors[nbr_comm]["score"]

				new_colors = [partition_colors[nbr_comm][col]+curr_colors[col_ind] for col_ind,col in enumerate(color_list)]
				# Calculate fairness score for newly created community (plus u)
				new_min_fair=1.0
				new_comm_len=sum(new_colors)
				for col_ind,_col in enumerate(color_list):
					if new_colors[col_ind]==0:
						new_min_fair=0.0 
					else:
						new_col_score=0.0
						if new_comm_len>new_colors[col_ind]:
							new_col_score=new_colors[col_ind]/(new_comm_len-new_colors[col_ind])
						# Set to min if less than current
						if new_col_score<new_min_fair:
							new_min_fair=new_col_score
				# Weigh by community length ratio for new_fair. Times Kcols
				new_fair=(K_cols-1) * new_min_fair * new_comm_len / n

				# Overall fairness score gain
				# new_fair (nbr) + post_fair (comms(u)) - new_score (nbr) - curr_fair (comms(u))
				fair_gain = (new_fair - new_score) + (post_fair - curr_fair)

				# Calculate new opt score.
				opt_gain = alpha * mod_gain + (1-alpha) * fair_gain

				# If opt gain tops previous gain found, set as new best
				if opt_gain > opt_best and (opt_gain - opt_best) > threshold:
					# # @DEBUG
					# print(f"Better found move: {u}->comm#{nbr_comm}")
					# print(f"u curr_f={curr_fair}, u post_f={post_fair}")
					# print(f"v curr_f={new_score}, v post_f={new_fair}")
					# print(f"fair gain={fair_gain}, mod_gain={mod_gain}")
					# print(f"opt gain={opt_gain}, (opt_best={opt_best})")
					# print("-----------")

					opt_best=opt_gain
					comm_best=nbr_comm

					new_fair_update=new_fair
					post_fair_update=post_fair


			# Then finalize move if necessary
			Stot[comm_best]+=degree
			if comm_best != comms[u]:
				# Finalize move

				# Get nodes in com
				com = G.nodes[u].get("nodes", {u})


				# Update partition, remove com nodes from comms(u)
				partition[comms[u]].difference_update(com)
				# Update inner partition, remove u from comms(u)
				inner_partition[comms[u]].remove(u)
				# Update partition, add com nodes to comm_best
				partition[comm_best].update(com)
				# Update inner partition, add u to comm_best
				inner_partition[comm_best].add(u)
				# Signal improvement
				improvement = True
					
				# Comment for infinite loop on a=0
				#n_moves += 1


				# Update new community colors
				for col_ind,col in enumerate(color_list):
					partition_colors[comm_best][col]+=curr_colors[col_ind]
				partition_colors[comm_best]["score"]=new_fair_update
				
				# Update old community colors
				for col_ind,col in enumerate(color_list):
					partition_colors[comms[u]][col]-=curr_colors[col_ind]
				partition_colors[comms[u]]["score"]=post_fair_update


				# Change new best community for u
				comms[u] = comm_best



	partition = list(filter(len, partition))
	inner_partition = list(filter(len, inner_partition))	
	partition_colors_new = list(filter(lambda ls: _full_partition_colors(ls,color_list), partition_colors))


	return partition, inner_partition, improvement, partition_colors_new



# ------------- HELPER FUNCTIONS ----------------

# Calculates average balance fairness of G for a given partition into communities
def fairness_base(G, partition, color_dist):
	colors=nx.get_node_attributes(G, "color")
	n=G.number_of_nodes()

	color_list=color_dist.keys()
	K_cols=len(color_list)

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

				# If any sum==0, or the sum of the color==len(ci): Leave balance to 0
				if sum_color==0 or sum_color==n_ci: 
					min_balance=0.0
					break

				# Otherwise: find if balance is min
				bal_score=sum_color/(n_ci-sum_color)
				if bal_score<min_balance:
					min_balance=bal_score

			# Set min_balance as the score. Normalize by K_cols s.t. max score is 1
			balance_ci=(K_cols-1) * min_balance * n_ci / n

			# Add to total sum (weighted by n_ci)
			sum_scores+=balance_ci
			F_dist.append(balance_ci)

	return sum_scores, F_dist



# Calculates average balance fairness for G with F_exp penalty
def fairness_fexp(G, partition, color_dist):
	colors=nx.get_node_attributes(G, "color")
	n=G.number_of_nodes()

	color_list=color_dist.keys()
	K_cols=len(color_list)

	# Calculate phi
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)


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
				sum_dist+=math.floor(color_dist[col]*n_ci/n)
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

				# If any sum==0, or the sum of the color==len(ci): Leave balance to 0
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


			# Add to total sum (weighted by n_ci)
			ci_score=(n_ci*fscore_ci)
			sum_scores+=ci_score
			F_dist.append(ci_score/n)

	return sum_scores/n, F_dist


# Helper function to remove all empty dicts
def _full_partition_colors(part_dict,color_list):
	# Iterate over all colors. If any of their sums is non-zero, return True
	for c in color_list:
		if part_dict[c]!=0: return True
	# Otherwise, if partition is empty: return False to discard.
	return False


# Get weights between node and its neighbor communities. Also for blue and red nodes
def neighbor_weights(nbrs, comms):
	weights=defaultdict(float)
	for nbr, w in nbrs.items():
		weights[comms[nbr]]+=w
	return weights
	

# Generate a new graph based on the partitions of a given graph.
# Also update partition colors
def _gen_graph(G, partition, colors):
	H = G.__class__()
	node2com = {}
	partition_colors = {}
	for i, part in enumerate(partition):
		nodes = set()
		for node in part:
			partition_colors[i]={}
			node2com[node] = i
			nodes.update(G.nodes[node].get("nodes", {node}))

		H.add_node(i, nodes=nodes)

	for node1, node2, wt in G.edges(data=True):
		wt = wt["weight"]
		com1 = node2com[node1]
		com2 = node2com[node2]
		temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
		H.add_edge(com1, com2, weight=wt + temp)
	return H, partition_colors



## -------------------------------------
##          Experiment logging
## -------------------------------------

def experiment(network, color_list=["blue","red"], n_reps=3, strategy=["base","step2","fexp","hybrid"], debug_mode=False):
	net=None
	# Load file
	with open(f"{obj_path}/{network}.nx","rb") as g_open:
		net=pickle.load(g_open)

	print(f"Network object {network} loaded.")
	print(f"{network}: N={net.number_of_nodes()}, M={net.number_of_edges()}")

	colors=nx.get_node_attributes(net, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in net.nodes():
		color_dist[colors[n_ind]]+=1

	print("Color distribution calculated.")
	for color in color_dist:
		print(f"{color}:{color_dist[color]}")


	
	# Run algorithm
	# alpha=[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,
	# 	0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]

	alpha=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	
	#alpha=[1.0]
	#alpha=[0.5,0.6,0.7]
	#alpha=[0.1]
	
	
	# Lists to keep stats for dataframe
	strat_l=[]
	alpha_l=[]
	time_l=[]
	time_s=[]
	ncomms_l=[]
	ncomms_s=[]
	mod_l=[]
	mod_s=[]
	fairbal_l=[]
	fairbal_s=[]
	fairexp_l=[]
	fairexp_s=[]

	# For each strategy:
	for strat in strategy:
		print(f"------------------\nSTRATEGY={strat}")

		# Run fair-balance on various values of alpha
		for a in alpha:
			print(f"-----------------\nAlpha={a}\n-----------------")

			modularity_list=[]
			fairnessbal_list=[]
			fairnessexp_list=[]
			ncomms_list=[]
			time_list=[]

			# and n reps
			for i in range(n_reps):
				if n_reps<=10:
					print(f"Run #{(i+1)}")
				else:
					if i%1000==0:
						print(f"Run #{(i+1)}/{n_reps}")

				start_time=time.time()
				res = fair_louvain_communities(net, color_list=color_list, alpha=a, strategy=strat)
				end_time=time.time()

				# # For debug: print also communities found
				# if debug_mode:
				# 	print("Communities obtained:")
				# 	print(res)

				# Track number of communities
				ncomms_list.append(len(res))

				# Track execution time
				time_list.append(end_time-start_time)

				# Track modularity score
				mod_overall = modularity(net, res, weight="weight")
				modularity_list.append(mod_overall)


				# Track both fairness scores
				F_bal_overall, F_bal_dist = fairness_base(net, res, color_dist)
				F_exp_overall, F_exp_dist = fairness_fexp(net, res, color_dist)
				fairnessbal_list.append(F_bal_overall)
				fairnessexp_list.append(F_exp_overall)

			

			# At end of reps: append to lists
			strat_l.append(strat)
			alpha_l.append(a)
			time_l.append(statistics.fmean(time_list))
			time_s.append(statistics.stdev(time_list))
			mod_l.append(statistics.fmean(modularity_list))
			mod_s.append(statistics.stdev(modularity_list))
			fairbal_l.append(statistics.fmean(fairnessbal_list))
			fairbal_s.append(statistics.stdev(fairnessbal_list))
			fairexp_l.append(statistics.fmean(fairnessexp_list))
			fairexp_s.append(statistics.stdev(fairnessexp_list))
			ncomms_l.append(statistics.fmean(ncomms_list))
			ncomms_s.append(statistics.stdev(ncomms_list))


	# Results dataframe
	df=pd.DataFrame.from_dict(
		{
			"strategy":strat_l,
			"alpha":alpha_l,
			"time":time_l,
			"time_std":time_s,
			"modularity":mod_l,
			"modularity_std":mod_s,
			"fair_bal":fairbal_l,
			"fair_bal_std":fairbal_s,
			"fair_exp":fairexp_l,
			"fair_exp_std":fairexp_s,
			"ncomms":ncomms_l,
			"ncomms_std":ncomms_s
		}, orient="columns"
	)

	# If not in debug mode: save to logfile
	if not debug_mode:
		# At end of run: create big dataframe for run stats and save to file
		df.to_csv(f"{log_path}/{network}.csv",index=False)
	# Otherwise simply print results to screen
	else:
		print(df[["strategy","alpha","modularity","fair_bal","fair_exp","ncomms"]])




	
# (Kobbie) Main(oo) --how is he doing these days?
"""
Arguments: python falcon_src.py network [color_list] [n_reps] [strat_list] [debug]
	- network: the network object to be run
	- color_list: (optional) the colors coded in the network object. Useful especially
				  for networks with multiple colours, otherwise defaults to red,blue
				  Should be input without quotes or spaces, separated by commas.
	- n_reps:	  (optional) the number of iterations for each run of alpha. Default 3
	- strat_list: (optional) the different optimization strategies to be tried for 
				  each run. Defaults to all (base,step2,fexp,hybrid). Similar formatting 
				  to color_list: no spaces, comma-separated values.
	- debug:	  (optional) runs in debug mode (i.e. no log file, but results printed
				  on screen) if the last argument is "debug"

Example call: python3 falcon_src.py facebook red,blue 100 base,step2 debug
	Will run the facebook network for colors R&B for 100 iterations, using the base 
	and step2 strategies, in debug mode.

"""
def main():
	args=sys.argv[1:]
	if len(args)==1:
		experiment(args[0])
	elif len(args)==2:
		clist=args[1].split(",")
		experiment(args[0], clist)
	elif len(args)==3:
		clist=args[1].split(",")
		experiment(args[0], clist, n_reps=int(args[2]))
	elif len(args)==4:
		clist=args[1].split(",")
		stratlist=args[3].split(",")
		experiment(args[0], clist, n_reps=int(args[2]), strategy=stratlist)
	elif len(args)==5:
		clist=args[1].split(",")
		stratlist=args[3].split(",")
		if args[4]=="debug":
			experiment(args[0], clist, n_reps=int(args[2]), strategy=stratlist, debug_mode=True)
		else:
			experiment(args[0], clist, n_reps=int(args[2]), strategy=stratlist)
	else:
		print("Wrong number of arguments: expected 1-4.")
		print("Usage: python falcon_src.py network [color_list] [n_reps] [strat_list] [debug]")

# python3 fairmod_src.py proximity_comb red_M,red_F,blue_M,blue_F,green_M,green_F,orange_M,orange_F 10 base debug
# python3 fairmod_src.py facebook_107_comb red_A,red_B,red_C,blue_A,blue_B,blue_C 10 base debug



if __name__ == '__main__':
	main()
