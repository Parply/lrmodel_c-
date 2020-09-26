import numpy as np

if __name__ == "__main__":
    data = np.genfromtxt("creditcard.csv",dtype=np.float64,delimiter=",",skip_header=1)
    x= data[:,:-1]
    y=data[:,-1]
    xt = x.transpose()
    xx= np.linalg.inv(np.matmul(xt,x))
    xxx = np.matmul(xx,xt)
    beta = np.matmul(xxx,y[:,None])
    pred = np.matmul(x,beta)
