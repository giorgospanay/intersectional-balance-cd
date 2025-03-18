import networkx as nx
import pandas as pd
import statsmodels.api as sm
import numpy as np
import pickle
import math

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Globals for paths
obj_path="../data/obj"
log_path="../logs"
plot_path="../plots"



def community_glm(network,algo="",dem_list=[]):
	# Read network
	with open(f"{obj_path}/{network}.nx","rb") as g_open:
		G=pickle.load(g_open)
	print(f"Network object {network} loaded.")
	print(f"{network}: N={G.number_of_nodes()}, M={G.number_of_edges()}")

	# Read partition
	with open(f"{log_path}/{network}_communities.dict","rb") as log_out:
		communities=pickle.load(log_out)
	print("Communities read from log_path.")

	# Read distributions
	with open(f"{log_path}/{network}_full_color_dist.dict","rb") as log_in:
		full_color_dist=pickle.load(log_in)
	with open(f"{log_path}/{network}_dem_color_dist.dict","rb") as log_in:
		dem_color_dist=pickle.load(log_in)
	print("Distributions read from log_path.")



	# Start with partition dictionary. Partition is list of lists (?)
	partition=communities[algo].communities

	# Convert partition dictionary to df
	node_community={}
	for c_id,nodes in enumerate(partition):
		for node in nodes:
			node_community[node]=c_id
	community_df=pd.DataFrame.from_dict(node_community,orient="index",columns=["community"])
	community_df.index.name="node"


	# Extract node attributes for selected demographic into df
	colors={dem:nx.get_node_attributes(G,dem) for dem in dem_list}
	node_data=pd.DataFrame(colors)
	node_data.index.name="node"
	# Merge community ids and node attributes
	df=community_df.join(node_data)



	# Encode categorical features if any
	df = df.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)

	# Separate features and target
	X = df.drop(columns=['community'])  # Features (node attributes)
	y = df['community']  # Target (community label)

	print(df)

	# Standardize features
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	print(X_scaled)

	# Train a logistic regression model
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
	model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
	model.fit(X_train, y_train)

	# Evaluate the model
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(f'Accuracy: {accuracy:.4f}')

	# Check feature importance
	feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
	print(feature_importance.sort_values(by='Coefficient', ascending=False))


	# # Encode categorical variables
	# df=pd.get_dummies(df,drop_first=True)
	# df=df.apply(pd.to_numeric).astype(float)
	# print(df)
	# # Dependent var: community id. Independent var: node attributes
	# y=df["community"]
	# X=df.drop(columns=["community"])
	# # GLM model to a multinomial distribution: predict community membership based
	# #   on attributes
	# X=sm.add_constant(X)  # Add constant term for intercept
	# glm=sm.GLM(y,X,family=sm.families.Binomial())
	# result=glm.fit()
	# # X = sm.add_constant(X)  # Add constant term for intercept
	# # multinomial_model = sm.MNLogit(y, X)  # Multinomial Logistic Regression
	# # result = multinomial_model.fit(method='newton', maxiter=100)
	# print(result.summary())


	# Add intercept to X
	X_with_const = sm.add_constant(X_scaled)

	# Fit a multinomial logistic regression model
	model = sm.MNLogit(y, X_with_const)  # Multinomial logistic regression
	result = model.fit()

	print(result.summary())


	return



def main():
	#community_glm("facebook_full",algo="louvain",dem="gender")
	community_glm("facebook_full",algo="louvain",dem_list=["gender","education"])
	

if __name__ == '__main__':
	main()

