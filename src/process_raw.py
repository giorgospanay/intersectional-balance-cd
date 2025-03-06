import networkx as nx
import pickle

raw_path="../data/raw"
obj_path="../data/obj"


def process_proximity_full():
	# Edgelist: edges.csv, color: nodes.csv
	edges_lines=[]
	color_lines=[]

	with open(f"{raw_path}/proximity/edges.csv","r") as f_edges:
		edges_lines=[line.rstrip().split(",") for line in f_edges]

	with open(f"{raw_path}/proximity/nodes.csv","r") as f_color:
		color_lines=[line.rstrip().split(",") for line in f_color]


	# Create proximity networks (class - 4 colors)
	net_prox_c=nx.Graph()
	# Add nodes and colors:
	for i,ln in enumerate(color_lines):
		# Ignore header line: 
		## index, id, class, gender, _pos
		if i==0: continue

		n_ind=int(ln[0])
		n_class=ln[2]
		n_gender=ln[3]

		ng_char=""

		if "M" in n_gender:
			ng_char="M"
		else:
			ng_char="F"

		if "2BIO" in n_class:
			# Red: class = 2BIO
			net_prox_c.add_node(n_ind,color=f"red_{ng_char}",subject="red",gender=ng_char)
		elif "PC" in n_class:
			# Blue: class = PC
			net_prox_c.add_node(n_ind,color=f"blue_{ng_char}",subject="blue",gender=ng_char)
		elif "PSI" in n_class:
			# Green: class = PSI
			net_prox_c.add_node(n_ind,color=f"green_{ng_char}",subject="green",gender=ng_char)
		elif "MP" in n_class:
			# Orange: class = MP
			net_prox_c.add_node(n_ind,color=f"orange_{ng_char}",subject="orange",gender=ng_char)

	# Add edges:
	for i,ln in enumerate(edges_lines):
		# Ignore header line: 
		## source, target, time
		if i==0: continue

		n1_ind=int(ln[0])
		n2_ind=int(ln[1])

		# Add edges ignoring time. Ignore duplicate edges
		if not net_prox_c.has_edge(n1_ind,n2_ind):
			net_prox_c.add_edge(n1_ind,n2_ind,weight=1.0)


	# Save object
	with open(f"{obj_path}/proximity_full.nx","wb") as tw_out:
		pickle.dump(net_prox_c,tw_out)

def process_facebook_full():
	# Edgelist: facebook_combined.txt
	# Features: all files *.feat (names in .names)

	edges_lines=[]

	with open(f"{raw_path}/facebook/facebook_combined.txt","r") as f_edges:
		edges_lines=[line.rstrip().split(" ") for line in f_edges]


	# Create facebook network
	net_fb=nx.Graph()

	# 0.feat-- gender 77,78 / education 53-55
	with open(f"{raw_path}/facebook/0.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[56])==1:
				education_type="C"
			elif int(ln[55])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[78])==1:
				net_fb.add_node(0,color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(0,color=f"blue_{education_type}",gender="blue",education=education_type)
	with open(f"{raw_path}/facebook/0.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[56])==1:
				education_type="C"
			elif int(ln[55])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[78])==1:
				net_fb.add_node(int(ln[0]),color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(int(ln[0]),color=f"blue_{education_type}",gender="blue",education=education_type)

	## Open feature files manually. If first gender attb==1, make red
	# 107.feat-- gender 264-265, education type 220-222
	with open(f"{raw_path}/facebook/107.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[223])==1:
				education_type="C"
			elif int(ln[222])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[265])==1:
				net_fb.add_node(107,color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(107,color=f"blue_{education_type}",gender="blue",education=education_type)
	with open(f"{raw_path}/facebook/107.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[223])==1:
				education_type="C"
			elif int(ln[222])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[265])==1:
				net_fb.add_node(int(ln[0]),color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(int(ln[0]),color=f"blue_{education_type}",gender="blue",education=education_type)

	# 348 -- gender 86,87 / education 59-61
	with open(f"{raw_path}/facebook/348.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[62])==1:
				education_type="C"
			elif int(ln[61])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[87])==1:
				net_fb.add_node(348,color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(348,color=f"blue_{education_type}",gender="blue",education=education_type)
	with open(f"{raw_path}/facebook/348.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[62])==1:
				education_type="C"
			elif int(ln[61])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[87])==1:
				net_fb.add_node(int(ln[0]),color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(int(ln[0]),color=f"blue_{education_type}",gender="blue",education=education_type)

	# 414 -- gender 63,64 / education 44-46
	with open(f"{raw_path}/facebook/414.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[47])==1:
				education_type="C"
			elif int(ln[46])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[64])==1:
				net_fb.add_node(414,color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(414,color=f"blue_{education_type}",gender="blue",education=education_type)
	with open(f"{raw_path}/facebook/414.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[47])==1:
				education_type="C"
			elif int(ln[46])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[64])==1:
				net_fb.add_node(int(ln[0]),color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(int(ln[0]),color=f"blue_{education_type}",gender="blue",education=education_type)

	# 686 -- gender 41,42 / education 21-23
	with open(f"{raw_path}/facebook/686.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[24])==1:
				education_type="C"
			elif int(ln[23])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[42])==1:
				net_fb.add_node(686,color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(686,color=f"blue_{education_type}",gender="blue",education=education_type)
	with open(f"{raw_path}/facebook/686.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[24])==1:
				education_type="C"
			elif int(ln[23])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[42])==1:
				net_fb.add_node(int(ln[0]),color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(int(ln[0]),color=f"blue_{education_type}",gender="blue",education=education_type)

	# 698 -- gender 26,27 / education 12-14
	with open(f"{raw_path}/facebook/698.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[15])==1:
				education_type="C"
			elif int(ln[14])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[27])==1:
				net_fb.add_node(698,color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(698,color=f"blue_{education_type}",gender="blue",education=education_type)
	with open(f"{raw_path}/facebook/698.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[15])==1:
				education_type="C"
			elif int(ln[14])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[27])==1:
				net_fb.add_node(int(ln[0]),color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(int(ln[0]),color=f"blue_{education_type}",gender="blue",education=education_type)

	# 1684 -- gender 147,148 / education 101-103
	with open(f"{raw_path}/facebook/1684.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[104])==1:
				education_type="C"
			elif int(ln[103])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[148])==1:
				net_fb.add_node(1684,color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(1684,color=f"blue_{education_type}",gender="blue",education=education_type)
	with open(f"{raw_path}/facebook/1684.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[104])==1:
				education_type="C"
			elif int(ln[103])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[148])==1:
				net_fb.add_node(int(ln[0]),color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(int(ln[0]),color=f"blue_{education_type}",gender="blue",education=education_type)

	# 1912 -- gender 259,260 / education 218-220
	with open(f"{raw_path}/facebook/1912.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[221])==1:
				education_type="C"
			elif int(ln[220])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[260])==1:
				net_fb.add_node(1912,color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(1912,color=f"blue_{education_type}",gender="blue",education=education_type)
	with open(f"{raw_path}/facebook/1912.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[221])==1:
				education_type="C"
			elif int(ln[220])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[260])==1:
				net_fb.add_node(int(ln[0]),color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(int(ln[0]),color=f"blue_{education_type}",gender="blue",education=education_type)

	# 3437 -- gender 117,118 / education 82-84
	with open(f"{raw_path}/facebook/3437.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[85])==1:
				education_type="C"
			elif int(ln[84])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[118])==1:
				net_fb.add_node(3437,color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(3437,color=f"blue_{education_type}",gender="blue",education=education_type)
	with open(f"{raw_path}/facebook/3437.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[85])==1:
				education_type="C"
			elif int(ln[84])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[118])==1:
				net_fb.add_node(int(ln[0]),color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(int(ln[0]),color=f"blue_{education_type}",gender="blue",education=education_type)

	# 3980 -- gender 19,20 / education 5-7
	with open(f"{raw_path}/facebook/3980.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[8])==1:
				education_type="C"
			elif int(ln[7])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[20])==1:
				net_fb.add_node(3980,color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(3980,color=f"blue_{education_type}",gender="blue",education=education_type)
	with open(f"{raw_path}/facebook/3980.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			# Get education type A/B/C
			if int(ln[8])==1:
				education_type="C"
			elif int(ln[7])==1:
				education_type="B"
			else: 
				education_type="A"

			if int(ln[20])==1:
				net_fb.add_node(int(ln[0]),color=f"red_{education_type}",gender="red",education=education_type)
			else:
				net_fb.add_node(int(ln[0]),color=f"blue_{education_type}",gender="blue",education=education_type)

	
	# Add edges
	for ln in edges_lines:
		net_fb.add_edge(int(ln[0]),int(ln[1]),weight=1.0)

	# Save object
	with open(f"{obj_path}/facebook_full.nx","wb") as fb_out:
		pickle.dump(net_fb,fb_out)







if __name__ == '__main__':
	### Processing raw data to NetworkX objects with node color attributes
	# process_proximity_comb()
	# process_facebook_107_comb()

	# process_proximity_dem()
	# process_facebook_107_dem()

	process_proximity_full()
	process_facebook_full()


