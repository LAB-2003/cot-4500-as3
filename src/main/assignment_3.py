import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

#1
def function(t,y):
    return t - (y**2)

def euler(t0,y0,tn,n):
    h=(tn-t0)/n

    for i in range(n):
        slope = function(t0,y0)
        yn = y0 + h *slope
        
        y0 = yn
        t0 = t0+h
    print("%.5f"%yn,"\n")

t0 = 0
y0 = 1
xn = 2
step = 10
euler(t0,y0,xn,step)

#2
def f(t,y):
    return t-(y**2)


def rk(t0,y0,tn,n):
    
    
    h = (tn-t0)/n
    
    for i in range(n):
        k1 = h * (f(t0, y0))
        k2 = h * (f((t0+h/2), (y0+k1/2)))
        k3 = h * (f((t0+h/2), (y0+k2/2)))
        k4 = h * (f((t0+h), (y0+k3)))
        k = (k1+2*k2+2*k3+k4)/6
        yn = y0 + k
        
        y0 = yn
        t0 = t0+h
    
    print("%.5f"%yn,"\n")

t0 = 0
y0 = 1
xn = 2
step = 10
rk(t0,y0,xn,step)

#3
import sys

# Reading number of unknowns
n = 3


a = np.zeros((n,n+1))


x = np.zeros(n)

# Reading augmented matrix coefficients
a = np.array([[2,-1,1,6],[1,3,1,0],[-1,5,4,-3]],dtype=np.double)

# Applying Gauss Elimination
for i in range(n):
    if a[i][i] == 0.0:
        sys.exit('Divide by zero detected!')
        
    for j in range(i+1, n):
        ratio = a[j][i]/a[i][i]
        
        for k in range(n+1):
            a[j][k] = a[j][k] - ratio * a[i][k]

# Back Substitution
x[n-1] = a[n-1][n]/a[n-1][n-1]

for i in range(n-2,-1,-1):
    x[i] = a[i][n]
    
    for j in range(i+1,n):
        x[i] = x[i] - a[i][j]*x[j]
    
    x[i] = x[i]/a[i][i]

# Displaying solution

print('[%0.0f.'%(x[0]),'%0.0f.'%(x[1]),'%0.0f.]\n'%(x[2]))


#4
def plu(A):
    
    #Get the number of rows
    n = A.shape[0]
    
    #Allocate space for P, L, and U
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    P = np.eye(n, dtype=np.double)
    
    #Loop over rows
    for i in range(n):
        
        #Permute rows if needed
        for k in range(i, n): 
            if ~np.isclose(U[i, i], 0.0):
                break
            U[[k, k+1]] = U[[k+1, k]]
            P[[k, k+1]] = P[[k+1, k]]
            
        #Eliminate entries below i with row 
        #operations on U and #reverse the row 
        #operations to manipulate L
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] = U[i+1:] - factor[:, np.newaxis] * U[i]
    m = np.linalg.det(A)
    print("%.5f"%m,"\n")
    print(L,"\n")
    print(U,"\n")
    
    
m = np.array([[1,1,0,3],[2,1,-1,1],[3,-1,-1,2],[-1,2,3,-1]],dtype=np.double)
plu(m)

#5

def isDDM(m, n) :
 
    # for each row
    for i in range(0, n) :        
     
        # for each column, finding
        # sum of each row.
        sum = 0
        for j in range(0, n) :
            sum = sum + abs(m[i][j])    
 
        # removing the
        # diagonal element.
        sum = sum - abs(m[i][i])
 
        # checking if diagonal
        # element is less than
        # sum of non-diagonal
        # element.
        if (abs(m[i][i]) < sum) :
            return False
 
    return True
 
# Driver Code
n = 5
m = [[ 9,0,5,2,1 ],
    [ 3,9,1,2,1],
    [ 0,1,7,2,3],
    [ 4,2,3,12,2],
    [ 3,2,4,0,8]]
 
if((isDDM(m, n))) :
    print ("True")
else :
    print ("False \n")

#6

def is_pos_def(x):
    print( np.all(np.linalg.eigvals(x) > 0))

x = [[2,2,1],
     [2,3,0],
     [1,0,2]]

is_pos_def(x)





