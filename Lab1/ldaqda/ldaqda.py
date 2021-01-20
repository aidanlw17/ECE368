import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
    
    N = x.shape[0]
    
    male_points = x[y==1]
    female_points = x[y==2]
    
    mu = np.mean(x, 0)
    mu_male = np.mean(male_points, 0)
    mu_female = np.mean(female_points, 0)
    
        
    mu = mu.reshape(-1, 1)
    mu_male = mu_male.reshape(-1, 1)
    mu_female = mu_female.reshape(-1, 1)
            
    male_count = male_points.shape[0]
    female_count = female_points.shape[0]
    
                    
    cov = np.zeros((2, 2))
    cov_male = np.zeros((2, 2))
    cov_female = np.zeros((2, 2))
                            
    for i in range(0, male_count):
        val = male_points[i]
        val = val.reshape(-1, 1)
        cov_male += np.matmul(val - mu_male, np.transpose(val - mu_male))
        
    for i in range(0, female_count):
        val = female_points[i]
        val = val.reshape(-1, 1)
        cov_female +=  np.matmul(val - mu_female, np.transpose(val - mu_female))
            
        
    cov = 1/(N-2) * (cov_male + cov_female)
        
    cov_male = 1/male_count * cov_male
    
    cov_female = 1/female_count * cov_female
        
        
    ''' 
    PLOTS BELOW
    '''
    
    # LDA
    plt.rcParams["figure.figsize"] = (10,10)
   
           
    
    plt.scatter(male_points[:male_count, 0], male_points[:male_count, 1], c = 'b')
    plt.scatter(female_points[:female_count, 0], female_points[:female_count, 1], c = 'r')


    axes = plt.gca()
    axes.set_xlim([50, 80])
    axes.set_ylim([80, 280])
    
    x_p = np.arange(50, 80, 1)
    y_p = np.arange(80, 280, 1)
    X, Y = np.meshgrid(x_p, y_p)
            
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    X_flat = X_flat.reshape(-1, 1)
    Y_flat = Y_flat.reshape(-1, 1)
    
    vec = np.concatenate((X_flat, Y_flat), axis = 1)
                        
    Z_male = util.density_Gaussian(np.transpose(mu_male)[0], cov, vec)
    Z_female = util.density_Gaussian(np.transpose(mu_female)[0], cov, vec)
    
    
    Z_male = Z_male.reshape((200, 30))
    Z_female = Z_female.reshape((200, 30))
        

    plt.contour(X, Y, Z_male, colors = 'blue')
    plt.contour(X, Y, Z_female, colors = 'red')
    
    # boundary calculation
    inv_sigma = np.linalg.inv(cov)
    
    lda_male = np.transpose(np.matmul(inv_sigma, mu_male))
    lda_male = np.matmul(lda_male, np.transpose(vec))
    lda_male = lda_male - 1/2 * np.matmul(np.matmul(np.transpose(mu_male), inv_sigma), mu_male)
    
    lda_female = np.transpose(np.matmul(inv_sigma, mu_female))
    lda_female = np.matmul(lda_female, np.transpose(vec))
    lda_female = lda_female - 1/2 * np.matmul(np.matmul(np.transpose(mu_female), inv_sigma), mu_female)    
    
    Z_bound = lda_male - lda_female
    
    Z_bound = Z_bound.reshape((200, 30))
    
    plt.contour(X, Y, Z_bound, 0)
    
    
    plt.xlabel("height")
    plt.ylabel("weight")
    plt.legend(["male", "female"], loc = "upper left")
    plt.show()
    
    ###QDA
    
    plt.scatter(male_points[:male_count, 0], male_points[:male_count, 1], c = 'b')
    plt.scatter(female_points[:female_count, 0], female_points[:female_count, 1], c = 'r')


    axes = plt.gca()
    axes.set_xlim([50, 80])
    axes.set_ylim([80, 280])
                    
    Z_male = util.density_Gaussian(np.transpose(mu_male)[0], cov_male, vec)
    Z_female = util.density_Gaussian(np.transpose(mu_female)[0], cov_female, vec)
    
    Z_male = Z_male.reshape((200, 30))
    Z_female = Z_female.reshape((200, 30))
    

    plt.contour(X, Y, Z_male, colors = 'blue')
    plt.contour(X, Y, Z_female, colors = 'red')
    
    # boundary calculation
    inv_sigma_male = np.linalg.inv(cov_male)
    det_sigma_male = np.linalg.det(cov_male)
    
    qda_male = np.zeros((vec.shape[0], 1))
    
    
    
    for i in range(0, vec.shape[0]):
        val = vec[i]
        val = val.reshape(-1, 1)
        inner_sub = np.subtract(val, mu_male)
        
        prod1 = np.matmul(inner_sub.transpose(), inv_sigma_male)
        qda_male[i] = -1/2 * np.matmul(prod1, inner_sub) - 1/2 * np.log(det_sigma_male)
        
        
        
        
    # qda female
    inv_sigma_female = np.linalg.inv(cov_female)
    det_sigma_female = np.linalg.det(cov_female)
    
    qda_female = np.zeros((vec.shape[0], 1))
    
    for i in range(0, vec.shape[0]):
        val = vec[i]
        val = val.reshape(-1, 1)
        inner_sub = np.subtract(val, mu_female)
        
        prod1 = np.matmul(inner_sub.transpose(), inv_sigma_female)
        qda_female[i] = -1/2 * np.matmul(prod1, inner_sub) - 1/2 * np.log(det_sigma_female)
        
    
    Z_qda = qda_male - qda_female
    
    Z_qda = (Z_qda.reshape((200, 30)))
    
    plt.contour(X, Y, Z_qda, 0)
    
    
    
    plt.xlabel("height")
    plt.ylabel("weight")
    plt.legend(["male", "female"], loc = "upper left")
    plt.show()    
    
    
     
    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
    
    # LDA
    inv_sigma = np.linalg.inv(cov)
    
    lda_male = np.transpose(np.matmul(inv_sigma, mu_male))
    lda_male = np.matmul(lda_male, np.transpose(x))
    lda_male = lda_male - 1/2 * np.matmul(np.matmul(np.transpose(mu_male), inv_sigma), mu_male)
    
    lda_female = np.transpose(np.matmul(inv_sigma, mu_female))
    lda_female = np.matmul(lda_female, np.transpose(x))
    lda_female = lda_female - 1/2 * np.matmul(np.matmul(np.transpose(mu_female), inv_sigma), mu_female)    
    
    Z = lda_male - lda_female
        
    Z[Z > 0] = 1
    Z[Z < 0] = 2
        
    N = x.shape[0]
        
    misclassified = (Z != y)*1
    
    misclassified = np.sum(misclassified) / N

    mis_lda = misclassified
    
    
    #QDA
    inv_sigma_male = np.linalg.inv(cov_male)
    det_sigma_male = np.linalg.det(cov_male)
    
    qda_male = np.zeros((N, 1))
    
    for i in range(0, N):
        val = x[i]
        val = val.reshape(-1, 1)
        inner_sub = np.subtract(val, mu_male)
        
        prod1 = np.matmul(inner_sub.transpose(), inv_sigma_male)
        qda_male[i] = -1/2 * np.matmul(prod1, inner_sub) - 1/2 * np.log(det_sigma_male)
        
    # qda female
    inv_sigma_female = np.linalg.inv(cov_female)
    det_sigma_female = np.linalg.det(cov_female)
    
    qda_female = np.zeros((N, 1))
    
    for i in range(0, N):
        val = x[i]
        val = val.reshape(-1, 1)
        inner_sub = np.subtract(val, mu_female)
        
        prod1 = np.matmul(inner_sub.transpose(), inv_sigma_female)
        qda_female[i] = -1/2 * np.matmul(prod1, inner_sub) - 1/2 * np.log(det_sigma_female)
    
    Z_qda = qda_male - qda_female
    
    Z_qda[Z_qda > 0] = 1
    Z_qda[Z_qda < 0] = 2
    
    Z_qda = Z_qda.transpose()
        
    N = x.shape[0]
        
    misclassified = (Z_qda != y)*1
    
    misclassified = np.sum(misclassified) / N

    mis_qda = misclassified
    
   
    
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    

    
    
    

    
