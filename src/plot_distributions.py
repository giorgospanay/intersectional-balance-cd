import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl

import pickle

# Globals for paths
obj_path="../data/obj"
log_path="../logs"
plot_path="../plots"

# @TODO: manually add values for heatmap colors?
# Load full color distribution for network from log_path, plot heatmap
def plot_fig1(network,plot_title="",palette={}):
	# Load color distribution
	with open(f"{log_path}/{network}_full_color_dist.dict","rb") as log_in:
		full_color_dist=pickle.load(log_in)
	print("Distribution loaded from file.")

	# Parse the dictionary into a DataFrame
	data=[]
	for key,value in full_color_dist.items():
		color1,color2=key.split('_')
		data.append([color1,color2,value])

	df=pd.DataFrame(data,columns=['color1','color2','value'])

	# Pivot the DataFrame to create a matrix format for the heatmap
	heatmap_data=df.pivot(index='color1',columns='color2',values='value')

	# Plot heatmap
	plt.figure(figsize=(6,4))
	ax=sns.heatmap(heatmap_data,annot=True,cmap="viridis",linewidths=0.5,fmt="d")

	# Set axis labels and title
	dem_list=list(palette.keys())
	ax.set_xlabel(f"{dem_list[1]}")
	ax.set_ylabel(f"{dem_list[0]}")
	plt.title(f"{plot_title}")

	# Save figure and show
	plt.savefig(f"{plot_path}/fig1_{network}.png",dpi=300,bbox_inches="tight")
	plt.show()

	return



def plot_fig2(network,plot_title="",algo_list=[],palette={},vmax=600):
	# Open pickle file with fairness scores
	with open(f"{log_path}/{network}_scores.dict","rb") as log_out:
		fairness_scores=pickle.load(log_out)
	print("Scores log loaded from file.")

	# Set large figure
	nrows=1
	ncols=len(algo_list)

	fig,axes=plt.subplots(nrows=nrows,ncols=ncols,figsize=(6*ncols,5*nrows))	

	for alg_i,algo in enumerate(algo_list):
		ax=axes[alg_i]

		d={"size":fairness_scores[algo]["community_sizes"],
			"score":fairness_scores[algo]["weighted_balance_dist"],
			"combo_score":fairness_scores[algo]["full_balance_dist"]}
		dem_list=list(palette.keys())

		# Convert dictionary to df. Also add intersectional combinations
		df=pd.DataFrame({
			"size":d["size"]*3,
			"score":d["score"][dem_list[0]]+d["score"][dem_list[1]]+d["combo_score"],
			"category":[dem_list[0]]*len(d["size"])+[dem_list[1]]*len(d["size"])+["combined"]*len(d["size"])
		})


		# Box plot to show score distribution for each category
		sns.boxplot(data=df,x="category",y="score",width=0.6,showcaps=True,boxprops={'facecolor': 'lightgray'},fliersize=0,ax=ax)

		# Overlay scatter plot with community sizes
		scatter = sns.stripplot(data=df,x="category",y="score",hue="size", 
								size=5,palette="viridis",edgecolor="black", 
								linewidth=0.6,jitter=True,legend=False,ax=ax)

		# Colormap for community sizes
		norm=mpl.colors.Normalize(vmin=0,vmax=vmax)
		sm=mpl.cm.ScalarMappable(cmap="viridis",norm=norm)
		sm.set_array([])

		# Add colorbar
		cbar=fig.colorbar(sm,ax=ax,orientation="vertical",pad=0.02)
		cbar.set_label("Community size")

		# Labels and title
		ax.set_xlabel("Demographic")
		ax.set_ylabel("Balance score")
		
	fig.suptitle(f"{plot_title}")

	# Save figure and show
	fig.savefig(f"{plot_path}/fig2_{network}.png",dpi=300,bbox_inches="tight")
	plt.show()

	return


# Row 1: balance(dem2) vs balance(dem1), color heatmap on comm size.
# Row 2: Balance of intersectional groups vs comm size.

def plot_fig3(network,plot_title="",algo_list=[],palette={},vmax=600):
	# Open pickle file with fairness scores
	with open(f"{log_path}/{network}_scores.dict","rb") as log_out:
		fairness_scores=pickle.load(log_out)
	print("Scores log loaded from file.")


	# Get dictionary with colors
	dem_list=list(palette.keys())

	# Create large figure
	nrows=2
	ncols=len(algo_list)

	fig,axes=plt.subplots(nrows=nrows,ncols=ncols,figsize=(6*ncols,5*nrows))	
	
	for alg_i,algo in enumerate(algo_list):
		
		# Plot row 1
		ax=axes[0][alg_i]
		d={"size":fairness_scores[algo]["community_sizes"],
			"score1":fairness_scores[algo]["weighted_balance_dist"][dem_list[0]],
			"score2":fairness_scores[algo]["weighted_balance_dist"][dem_list[1]]}
		# Convert dictionary to DataFrame
		df=pd.DataFrame(d)
		# Normalize csize for colormap
		norm=mpl.colors.Normalize(vmin=0,vmax=vmax)
		cmap=plt.cm.viridis

		# Scatter plot with color based on csize
		scatter=ax.scatter(x=df["score1"],y=df["score2"],c=df["size"],cmap=cmap, 
							 s=100,edgecolors='black',linewidth=0.6)

		# Add colorbar for csize
		cbar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax)
		cbar.set_label("Community size")
		# Labels and title
		ax.set_xlabel(f"{dem_list[0]} balance")
		ax.set_ylabel(f"{dem_list[1]} balance")


		# Plot row 2
		ax=axes[1][alg_i]
		d={"size":fairness_scores[algo]["community_sizes"],
		"score":fairness_scores[algo]["full_balance_dist"]}
		# Convert dictionary to df
		df=pd.DataFrame(d)
		# Plot the point plot
		scatter=ax.scatter(x=df["size"],y=df["score"],s=100,edgecolors='black',linewidth=0.6)

		#ax=sns.scatterplot(data=df,x="size",y="score",hue="category",style="category",s=100,edgecolor="black")
		
		# Set labels and title
		ax.set_xlabel("Community size")
		ax.set_ylabel("Balance score")



	# Save and plot
	fig.suptitle(f"{plot_title}")
	fig.savefig(f"{plot_path}/fig3_{network}.png",dpi=300,bbox_inches="tight")
	plt.show()

	return


# Load stats, plot points: community size vs. balance score
def plot_fig4(network,plot_title="",algo_list=[],palette={}):
	# Open pickle file with fairness scores
	with open(f"{log_path}/{network}_scores.dict","rb") as log_out:
		fairness_scores=pickle.load(log_out)
	print("Scores log loaded from file.")


	nrows=1
	ncols=len(algo_list)

	fig,axes=plt.subplots(nrows=nrows,ncols=ncols,figsize=(6*ncols,5*nrows))	

	for alg_i,algo in enumerate(algo_list):
		ax=axes[alg_i]


		d={"size":fairness_scores[algo]["community_sizes"],"score":fairness_scores[algo]["weighted_balance_dist"]}
		dem_list=list(palette.keys())

		# Convert dictionary to df
		df=pd.DataFrame({
			"size":d["size"]*2,
			"score":d["score"][dem_list[0]]+d["score"][dem_list[1]],
			"category":[dem_list[0]]*len(d["size"])+[dem_list[1]]*len(d["size"])
		})

		# Plot the point plot
		sns.scatterplot(data=df,x="size",y="score",hue="category",style="category",s=100,edgecolor="black",ax=ax)

		# Set labels and title
		ax.set_xlabel("Community size")
		ax.set_ylabel("Balance score")
	
	fig.suptitle(f"{plot_title}")

	# Save figure and show
	fig.savefig(f"{plot_path}/fig4_{network}.png",dpi=300,bbox_inches="tight")
	plt.show()

	return



def main():
	algo_list=["louvain","infomap","sbm_dl"]

	facebook_palette={"gender":"#FF5733","education": "#3498DB"}
	facebook_vmax=600
	# Heatmap: intersectional groups
	plot_fig1("facebook_full",plot_title="",palette=facebook_palette)
	# Box plot: distribution of demographic balance (per algorithm)
	plot_fig2("facebook_full",plot_title="",algo_list=algo_list,palette=facebook_palette,vmax=facebook_vmax)
	# Box plot: distribution of demographic balance (per algorithm)
	plot_fig3("facebook_full",plot_title="",algo_list=algo_list,palette=facebook_palette,vmax=facebook_vmax)
	# Scatterplots: size vs balance (per algorithm)
	plot_fig4("facebook_full",plot_title="",algo_list=algo_list,palette=facebook_palette)


	twitch_palette={"maturity":"#FF5733","language": "#3498DB"}
	twitch_vmax=50000
	# Heatmap: intersectional groups
	plot_fig1("twitch_full",plot_title="",palette=twitch_palette)
	# Box plot: distribution of demographic balance (per algorithm)
	plot_fig2("twitch_full",plot_title="",algo_list=algo_list,palette=twitch_palette,vmax=twitch_vmax)
	# Box plot: distribution of demographic balance (per algorithm)
	plot_fig3("twitch_full",plot_title="",algo_list=algo_list,palette=twitch_palette,vmax=twitch_vmax)
	# Scatterplots: size vs balance (per algorithm)
	plot_fig4("twitch_full",plot_title="",algo_list=algo_list,palette=twitch_palette)



if __name__ == '__main__':
	main()