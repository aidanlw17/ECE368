import numpy as np
import matplotlib.pyplot as plt
import util

plt.rcParams["figure.figsize"] = (10,10)


def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    
    x_p = np.arange(-1, 1, 0.1)
    y_p = np.arange(-1, 1, 0.1)
    X, Y = np.meshgrid(x_p, y_p)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    X_flat = X_flat.reshape(-1, 1)
    Y_flat = Y_flat.reshape(-1, 1)
    
    vec = np.concatenate((X_flat, Y_flat), axis = 1)
        
    mu = np.array([0, 0])
    
    cov = np.array([[beta, 0], [0, beta]])
    
    a = util.density_Gaussian(mu, cov, vec)
        
    a = a.reshape((20, 20))
        
    plt.contour(X, Y, a)
    
    plt.scatter([-0.1], [-0.5])
    
    plt.title("p(a)")
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.savefig('prior.pdf')
    plt.show()
    
    
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    
    x_p = np.arange(-1, 1, 0.01)
    y_p = np.arange(-1, 1, 0.01)
    X, Y = np.meshgrid(x_p, y_p)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    X_flat = X_flat.reshape(-1, 1)
    Y_flat = Y_flat.reshape(-1, 1)
    
    vec = np.concatenate((X_flat, Y_flat), axis = 1)
    
    one = np.ones((x.shape[0], 1))
    
    x_1 = np.concatenate((one, x), axis=1)
                            
    mu_a = np.array([0, 0])
    
    cov_a = np.array([[beta, 0], [0, beta]])
    
    cov_ai = np.array([[1.0/beta, 0], [0, 1.0/beta]])
            
    a_map = np.matmul( np.linalg.inv(cov_ai + 1.0/sigma2 * np.matmul(x_1.T, x_1)), 1.0/sigma2 * np.matmul(x_1.T, z))
    
    sigma_az = np.linalg.inv(cov_ai + 1.0/sigma2 * np.matmul(x_1.T, x_1))
    
                           
    a = util.density_Gaussian(a_map.T[0], sigma_az, vec)

    a = a.reshape((x_p.shape[0], x_p.shape[0]))
        
    plt.contour(X, Y, a)
    
    plt.scatter([-0.1], [-0.5])
    
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.savefig('posterior100.pdf')

    plt.show()
    
    mu = a_map
    Cov = sigma_az
    
   
   
    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    x_p = np.arange(-4, 4, 0.01)
    y_p = np.arange(-4, 4, 0.01)
    X, Y = np.meshgrid(x_p, y_p)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    X_flat = X_flat.reshape(-1, 1)
    Y_flat = Y_flat.reshape(-1, 1)
    
    vec = np.concatenate((X_flat, Y_flat), axis = 1)
            
    x = np.asarray(x)
    
    x = x.reshape(x.shape[0], 1)
        
    one = np.ones((x.shape[0], 1))
    
    x_1 = np.concatenate((one, x), axis=1)
    
    mu_new = np.matmul(x_1, mu)
    
    cov_new = np.matmul( np.matmul(x_1, Cov), x_1.T) + sigma2
    
    axes = plt.gca()
    axes.set_xlim([-4, 4])
    axes.set_ylim([-4, 4])
    
    # part3 plot
    plt.scatter(x_train, z_train, color='red', s = 5)
    
    
    plt.errorbar(x = x, y = mu_new, yerr = np.sqrt(np.diagonal(cov_new)), marker='x', color='blue')
    
    plt.legend(["Training Samples", "Predicted Targets"])


    plt.xlabel("x")
    plt.ylabel("z")
    plt.savefig('predict100.pdf')
    plt.show()    
    
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
