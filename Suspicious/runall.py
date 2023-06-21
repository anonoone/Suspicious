import os

model=["SGCN","SSGC","SSage"]
model=["SGCN"]

#model=["CoLA"]
#error=[0,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#error=[1.0]
#data=["Cora"]
#data=["reddit"]	
#ano=["Con","Str","Min","All","Out","MinOnly"]
#data=["Cora","PubMed","Computers","Photo","Books","reddit"]
data=["Computers","Photo","Books"]

anor=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
ano=["MinOnly"]
for a in ano:
		for m in model:
				for d in data:
						for ar in anor:
								os.system("python cleanrun.py --dataset='"+d+"' --training='suspicious' --model='"+m+"' --anomaly='"+a+"' --anomalyrate='"+str(ar)+"'") 
data=["Cora","Computers","Photo"]

ano=["Out"]
for a in ano:
		for m in model:
				for d in data:
						for ar in anor:
								os.system("python cleanrun.py --dataset='"+d+"' --training='suspicious' --model='"+m+"' --anomaly='"+a+"' --anomalyrate='"+str(ar)+"'")