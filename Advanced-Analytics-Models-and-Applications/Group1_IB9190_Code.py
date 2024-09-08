# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:18:13 2024

@author: Belly
"""
#%%
import pandas as pd
import numpy as np
from pyomo.environ import *

#%% data import and parameter preparation
A = pd.read_excel("Group1_IB9190_Input_Data.xlsx",sheet_name="A",index_col=0)
# A: If student i should take class k

S = A.index 
# S: total amount of students in MSBA

C = np.array(A.columns)
# C: classes' names

A = np.array(A)

Ct = pd.read_excel("Group1_IB9190_Input_Data.xlsx",sheet_name="Ct")
Ct = np.array(Ct.iloc[:,3:])
# Ct: whether each class k is going on at each time t

M = 2 
# the number of groups taking turns to have in-person classes (can be changed in Q3)

p = 0.3
#p: porportion of social diatance capacity caused by COVID-19 (can be changed in Q3)

c = np.array(round(pd.read_excel("Group1_IB9190_Input_Data.xlsx",sheet_name="c",index_col=0) * p))
# c: social distance capacity for each class k

E = round(40 * p)
# E: the capacity for the excess room (we choose 2.007 in wbs and multiply it by the social distancing proportion)

Tao = np.array(range(167)) 
# Tao : from 0-167 (Sunday 0am to Monday Saturday 11pm)

lam = 0.25 
miu = 0
# according the paper, they are constant numbers

#%% built model
model = ConcreteModel()

model.pi = Var(range(len(S)),range(M),domain=Binary)

model.ejk = Var(range(M),range(len(C)),domain=NonNegativeIntegers)

model.deltajk = Var(range(M),range(len(C)), domain=NonNegativeReals)

model.s = Var(domain=NonNegativeReals)
 
model.stj = Var(range(len(Tao)), range(M), domain = NonNegativeIntegers)

model.obj = Objective(
    expr = sum(sum(model.ejk[j,k] for j in range(M)) for k in range(len(C)))
         + lam * sum(sum(model.deltajk[j,k] for j in range(M)) for k in range(len(C)))
         + miu * model.s
         , sense=minimize)

def rule1(model,i):
    return sum(model.pi[i,j] for j in range(M)) == 1
model.const1 = Constraint(range(len(S)),rule = rule1)

def rule2(model,j,k):
    return (sum(model.pi[i,j] for i in range(len(S)) if A[i,k]==1) - c[k]) <= model.ejk[j,k]
model.const2 = Constraint(range(M), range(len(C)), rule = rule2)

def rule3_1(model,j,k):
    return -1* model.deltajk[j,k] <= sum(model.pi[i,j] for i in range(len(S)) if A[i,k]==1) - (sum(A[:,k]) / M)
model.const3_1 = Constraint(range(M), range(len(C)), rule = rule3_1)

def rule3_2(model,j,k):
    return sum(model.pi[i,j] for i in range(len(S)) if A[i,k]==1) - (sum(A[:,k]) / M) <= model.deltajk[j,k]
model.const3_2 = Constraint(range(M), range(len(C)), rule = rule3_2)

def rule4(model,t,j):
    return model.s >= model.stj[t,j] - E
model.const4 = Constraint(range(len(Tao)), range(M), rule = rule4)

def rule5(model,t,j):
    return model.stj[t,j] >= sum(model.ejk[j,k] for k in range(len(C)) if Ct[t,k] ==1 )
model.const5 = Constraint(range(len(Tao)), range(M), rule = rule5)

solver = SolverFactory('gurobi')
results = solver.solve(model, tee = True)

if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    model.solutions.load_from(results)
else:
    print("Solve failed.")

print("The minimum objective value:",round(value(model.obj),2))

#%% show the results of decision variables
pi = np.zeros((len(S), M))
for i in range(len(S)):
    for j in range(M):
        pi[i,j] = model.pi[i,j].value
pi = pd.DataFrame(pi)
pi.index = S
pi.columns = range(M)
#print(pi)

ejk = np.zeros((M,len(C)))
for j in range(M):
    for k in range(len(C)):
        ejk[j,k] = model.ejk[j,k].value
ejk = pd.DataFrame(ejk)
ejk.index = range(M)
ejk.columns = C
TE = sum(sum(ejk.iloc[j,:]) for j in range(M))

deltajk = np.zeros((M,len(C)))
for j in range(M):
    for k in range(len(C)):
        deltajk[j,k] = model.deltajk[j,k].value
deltajk = pd.DataFrame(deltajk)
deltajk.index = range(M)
deltajk.columns = C
print(deltajk)
TD = round(sum(sum(deltajk.iloc[j,:]) for j in range(M)),2)

stj = np.zeros((len(Tao),M))
for t in range(len(Tao)):
    for j in range(M):
        stj[t,j] = model.stj[t,j].value
stj = pd.DataFrame(stj)
# print(stj)

SSE = max(max(stj)-round(E*p),0)
print(SSE)

#excel_file_path = 'results.xlsx'
#with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    #pi.to_excel(writer, sheet_name='pi', index=True)
    #ejk.to_excel(writer, sheet_name='ejk', index=True)
    #deltajk.to_excel(writer, sheet_name='deltajk', index=True)
    #stj.to_excel(writer, sheet_name='stj', index=True)

#%% Q3 try different parameters
df_results = pd.DataFrame()

def model_obj(P,M):
    
    model = ConcreteModel()

    model.pi = Var(range(len(S)),range(M),domain=Binary)

    model.ejk = Var(range(M),range(len(C)),domain=NonNegativeIntegers)

    model.deltajk = Var(range(M),range(len(C)), domain=NonNegativeReals)

    model.s = Var(domain=NonNegativeReals)
     
    model.stj = Var(range(len(Tao)), range(M), domain = NonNegativeIntegers)

    model.obj = Objective(
        expr = sum(sum(model.ejk[j,k] for j in range(M)) for k in range(len(C)))
             + lam * sum(sum(model.deltajk[j,k] for j in range(M)) for k in range(len(C)))
             + miu * model.s
             , sense=minimize)

    def rule1(model,i):
        return sum(model.pi[i,j] for j in range(M)) == 1
    model.const1 = Constraint(range(len(S)),rule = rule1)

    def rule2(model,j,k):
        return (sum(model.pi[i,j] for i in range(len(S)) if A[i,k]==1) - c[k]) <= model.ejk[j,k]
    model.const2 = Constraint(range(M), range(len(C)), rule = rule2)

    def rule3_1(model,j,k):
        return -1* model.deltajk[j,k] <= sum(model.pi[i,j] for i in range(len(S)) if A[i,k]==1) - (sum(A[:,k]) / M)
    model.const3_1 = Constraint(range(M), range(len(C)), rule = rule3_1)

    def rule3_2(model,j,k):
        return sum(model.pi[i,j] for i in range(len(S)) if A[i,k]==1) - (sum(A[:,k]) / M) <= model.deltajk[j,k]
    model.const3_2 = Constraint(range(M), range(len(C)), rule = rule3_2)

    def rule4(model,t,j):
        return model.s >= model.stj[t,j] - E
    model.const4 = Constraint(range(len(Tao)), range(M), rule = rule4)

    def rule5(model,t,j):
        return model.stj[t,j] >= sum(model.ejk[j,k] for k in range(len(C)) if Ct[t,k] ==1 )
    model.const5 = Constraint(range(len(Tao)), range(M), rule = rule5)

    solver = SolverFactory('gurobi')
    results = solver.solve(model, tee = True)

    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        model.solutions.load_from(results)
    else:
        print("Solve failed.")
        
    ejk = np.zeros((M,len(C)))
    for j in range(M):
        for k in range(len(C)):
            ejk[j,k] = model.ejk[j,k].value
    ejk = pd.DataFrame(ejk)
    ejk.index = range(M)
    ejk.columns = C
    TE = sum(sum(ejk.iloc[j,:]) for j in range(M))
        
    deltajk = np.zeros((M,len(C)))
    for j in range(M):
        for k in range(len(C)):
            deltajk[j,k] = model.deltajk[j,k].value
    deltajk = pd.DataFrame(deltajk)
    deltajk.index = range(M)
    deltajk.columns = C
    print(deltajk)
    TD = round(sum(sum(deltajk.iloc[j,:]) for j in range(M)),2)

    stj = np.zeros((len(Tao),M))
    for t in range(len(Tao)):
        for j in range(M):
            stj[t,j] = model.stj[t,j].value
    stj = pd.DataFrame(stj)
    # print(stj)

    SSE = max(max(stj)-round(E*p),0)
    
    print("The minimum objective value:",round(value(model.obj),2))
    
    print("SSE:",SSE)
        
    return P, M, round(value(model.obj),2),TE,TD,SSE

model_obj(0.3,3)

P = [0.3,0.25,0.2]
M = [2,3,4,5,6,7,8,9,10]

for p in P:
    for m in M:
        df_results = pd.concat([df_results, pd.DataFrame(np.array(model_obj(p,m))).T],axis=0)

df_results.columns = ["Social Distance Capacity","M","Objective Value","TE","TD","SSE"]
df_results.to_excel("Q3_results.xlsx")

