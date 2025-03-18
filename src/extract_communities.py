import networkx as nx
import cdlib
from cdlib.algorithms import louvain, leiden, walktrap, pycombo, infomap, sbm_dl
import pickle
import math

# Globals for paths
obj_path="../data/obj"
log_path="../logs"
plot_path="../plots"


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


## Experiment 

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

	# Save distributions to pickle logfile
	with open(f"{log_path}/{network}_full_color_dist.dict","wb") as log_out:
		pickle.dump(full_color_dist,log_out)
	with open(f"{log_path}/{network}_dem_color_dist.dict","wb") as log_out:
		pickle.dump(dem_color_dist,log_out)
	print("Distributions saved to log_path. \n")


	# Apply community detection algorithms
	print("Run community detection algorithms.")
	communities = {
		"louvain": run_louvain(G),
		#"leiden": run_leiden(G),
		#"walktrap": run_walktrap(G),
		#"pycombo": run_combo(G),
		"infomap": run_infomap(G),
		"sbm_dl": run_sbm_dl(G)
	}

	# Save communities detected as pickle file
	with open(f"{log_path}/{network}_communities.dict","wb") as log_out:
		pickle.dump(communities,log_out)
	print("Community dictionary saved to log_path.")


def main():
	#experiment("proximity_full",dem_list=["subject","gender"])
	experiment("facebook_full",dem_list=["gender","education"])
	#experiment("twitch_full",dem_list=["maturity","language"])


if __name__ == '__main__':
	main()


