import os

model=["SGCN","SSGC","SSage"]
error=[0,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
data=["Cora","PubMed","Computers","Photo"]
ano=["Out","MinOnly"]
for a in ano:
		for m in model:
				for d in data:
						for e in error:
								os.system("python cleanrun.py --dataset='"+d+"' --vsanomalyrate="+str(e)+" --training='suspicious' --model='"+m+"' --anomaly='"+a+"'") 
								model=["SGCN","SSGC","SSage"]
error=[0,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
data=["Books","reddit"]
ano=["MinOnly"]
for a in ano:
		for m in model:
				for d in data:
						for e in error:
								os.system("python cleanrun.py --dataset='"+d+"' --vsanomalyrate="+str(e)+" --training='suspicious' --model='"+m+"' --anomaly='"+a+"'") 

model=["DOMINANT","ANOMALOUS","AnomalyDAE"]
data=["Cora","PubMed","Computers","Photo"]
ano=["Out","MinOnly"]
for a in ano:
		for m in model:
				for d in data:
						os.system("python cleanrun.py --dataset='"+d+"' --training='suspicious' --model='"+m+"' --anomaly='"+a+"'")
model=["DOMINANT","ANOMALOUS","AnomalyDAE"]
data=["Books","reddit"]
ano=["MinOnly"]
for a in ano:
		for m in model:
				for d in data:
						os.system("python cleanrun.py --dataset='"+d+"' --training='suspicious' --model='"+m+"' --anomaly='"+a+"'")
