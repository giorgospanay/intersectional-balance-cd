import networkx as nx
import cdlib
from cdlib.algorithms import louvain, leiden, walktrap, pycombo, infomap, sbm_dl
import pickle
import math

# Globals for paths
obj_path="../data/obj"
log_path="../logs/"
plot_path="../plots/"

## -----------------------------------------------------------------------------
##                             CD algorithm calls
## -----------------------------------------------------------------------------

def run_louvain(G):
	print("Running Louvain")
	louvain_communities = louvain(G)
	return louvain_communities

def run_leiden(G):
	print("Running Leiden")
	leiden_communities = leiden(G)
	return leiden_communities

def run_walktrap(G):
	print("Running Walktrap")
	walktrap_communities = walktrap(G)
	return walktrap_communities

def run_combo(G):
	print("Running Combo")
	pycombo_communities = pycombo(G)
	return pycombo_communities

def run_infomap(G):
	print("Running InfoMap")
	infomap_communities = infomap(G)
	return infomap_communities

def run_sbm_dl(G):
	print("Running SBM-DL")
	sbmdl_communities = sbm_dl(G)
	return sbmdl_communities


## -----------------------------------------------------------------------------
##                             Fairness metrics
## -----------------------------------------------------------------------------


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




def experiment(network, dem_list=[]):
	# Load file
	with open(f"{obj_path}/{network}.nx","rb") as g_open:
		G=pickle.load(g_open)

	print(f"Network object {network} loaded.")
	print(f"{network}: N={G.number_of_nodes()}, M={G.number_of_edges()}")

	# Calculate color distributions for full intersectional groups, and per demographic
	colors=nx.get_node_attributes(G,"color")
	dem_colors={d:nx.get_node_attributes(G,d) for d in dem_list}
	full_color_dist={}
	dem_color_dist={d:{} for d in dem_list}
	for n_ind in G.nodes():
		# Get node groups
		n_col=colors[n_ind]
		n_dem={d:(dem_colors[d][n_ind]) for d in dem_list}

		# Add to full intersectional dict
		if n_col not in full_color_dist:
			full_color_dist[n_col]=1
		else:
			full_color_dist[n_col]+=1

		# Add to dict per demographic
		for d in dem_list:
			if n_dem[d] not in dem_color_dist[d]:
				dem_color_dist[d][n_dem[d]]=1
			else:
				dem_color_dist[d][n_dem[d]]+=1

	# @DEBUG: Print color distributions
	print("Full intersectional group node distribution:")
	for col in full_color_dist:
		print(f"{col}: {full_color_dist[col]}")
	print("-----------------------------------\n")
	print("Node distribution per demographic:")
	for d in dem_color_dist:
		print(f"---{d}---")
		for c in dem_color_dist[d]:
			print(f"{c}: {dem_color_dist[d][c]}")
	print("-----------------------------------\n")

	# Apply community detection algorithms
	print("Run community detection algorithms.")
	communities = {
		"louvain": run_louvain(G),
		"leiden": run_leiden(G),
		"walktrap": run_walktrap(G),
		"pycombo": run_combo(G),
		"infomap": run_infomap(G),
		"sbm_dl": run_sbm_dl(G)
	}

	# # Print detected communities for each algorithm
	# for algo, comm in communities.items():
	# 	print(f"{algo} detected {len(comm.communities)} communities.")
	# 	for i, c in enumerate(comm.communities):
	# 		print(f"  Community {i+1}: {c}")
	# 	print("-" * 50)

	# NMI, other community-wise quality metrics?

	# Run fairness calculations
	fairness_scores = {algo:{} for algo in communities}

	for algo in communities:

		print(f"{algo}")

		full_balance, F_dist = fairness_base(
			G,
			communities[algo].communities,
			full_color_dist
		) 

		overall_balance, overall_fexp, balance_scores, fexp_scores, phi_scores = weighted_intersectional_balance(
			G, 
			communities[algo].communities, 
			dem_color_dist,
			{d:1.0 for d in dem_list} # change later with arg
		)

		print(full_balance)
		print(F_dist)

		print(overall_balance)
		print(balance_scores)



	return


def main():
	#experiment("proximity_full",dem_list=["subject","gender"])
	experiment("facebook_full",dem_list=["gender","education"])


if __name__ == '__main__':
	main()


