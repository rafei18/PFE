from pulp import*
import numpy as np

def weight_derivation(A):
    
    rows_sum = np.sum(A, axis = 1)
    Sum = 0
    Max_r = Sum
    for i in range(0, A.shape[0]):
        Sum = 0
        for j in range(0, A.shape[1]):
            Sum = (Sum + A[i][j] * rows_sum[j])
        Sum = Sum / float(rows_sum[i])
        if(Sum > Max_r):
            Max_r = Sum
            
    #Matrix transpose of A
    
    A_trans = A.transpose()
    rows_sum = np.sum(A_trans, axis = 1) #columns of original matrix A
    Sum = 0
    Max_c = Sum
    for i in range(0, A_trans.shape[0]):
        Sum = 0
        for j in range(0, A_trans.shape[1]):
            Sum = (Sum + A_trans[i][j] * rows_sum[j])
        Sum = Sum / float(rows_sum[i])
        if(Sum > Max_c):
            Max_c = Sum
    Beta = min(Max_r, Max_c)
    print("Beta =", Beta)
    
    # parti pulp
    # initilaliser la classe
    model = LpProblem("minimize", LpMinimize)

    # define decision variables 
    m,n = A.shape
    vect_w = []
    vect_z = []
    for i in range(0,n):#n
    
        w = LpVariable('w%s'%i, lowBound=0,cat='Integer')
        vect_w.append(w)
        z = LpVariable('z%s'%i, lowBound=0,cat='Integer')
        vect_z.append(z)

    # objectif function
    model += lpSum(vect_z[i] for i in range(0,n)) #n
   

    # constraint
    # constraint 1
    for i in range(0,n): #n
        #model += lpSum([A[i,j]*vect_z[j] == vect_w[i] for j in range(0,3)])
        model += lpSum([A[i,j]*vect_z[j]  for j in range(0,n)]) == vect_w[i] #n
    
    #constraint 2
    for i in range(0,n): #n
        model += vect_z[i] - ((1/n)*vect_w[i]) <=0
    
    # constraint 3
    var = 1/Beta
    for i in range(0,n):
        model += vect_z[i] - (var*vect_w[i]) >=0      # ..... (1)
    
    #constraint 4    
    model += lpSum([vect_w[j] for j in range(0,n)]) == 1              # ...... (2)
    #model += lpSum(vect_w) == 1   
    
        
    model.writeLP("weight_derivation.lp")
    model.solve()
    for v in model.variables():
        print(v.name, "=", v.varValue)
    
    # parti CR
    j = value(model.objective)
    ci = (1-(n*j))/((n-1)*j)
    random_index = np.array([0.5247,0.8816,1.1086,1.2479,1.3417,1.4057,1.4499,1.4854]) # dans le pdf que vous m'avez envoyé 
    ri = random_index[n-3]
    print("ri = {}".format(ri))
    cr = ci/ri
    if (cr<0.1):
        print("cr = {}".format(cr))
        print("The pairwise comparison matrix is consistent")
    else:
        print("The pairwise comparison matrix is inconsistent")
        
