import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

import time


def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here
    
    """
    1. Open file
    2. Read word from file
    3. Update dictionary with occurence of each word
    4. Repeat until file complete
    5. Repeat for all given files
    6. Do this for both SPAM and HAM files
    7. Once finished parsing all files, update dictionary with probability (use variable to keep track of total # of words)
    
    """
    
    spam_list = file_lists_by_category[0]
    ham_list = file_lists_by_category[1]
    
    spam_dict = {}
    
    spam_dict = get_word_freq(spam_list)

    ham_dict = get_word_freq(ham_list)
    
    total_dict = {**spam_dict, **ham_dict}
    
    spam_d = len(spam_dict)
    
    ham_d = len(ham_dict)
    
    # dictionary
    total_d = len(total_dict)
    
    total_spam = sum(spam_dict.values())
    total_ham = sum(ham_dict.values())
            
    p_d = {}
    q_d = {}
    
    for i in total_dict:
        p_d[i] = (spam_dict[i] + 1) / (total_d + total_spam)
        
        
    for i in total_dict:
        q_d[i] = (ham_dict[i] + 1) / (total_d + total_ham)
        
        
    probabilities_by_category = (p_d, q_d)
    
    
    return probabilities_by_category



def classify_new_email(filename,probabilities_by_category,prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    
    # given filename, first need to parse the file and get all the words
    
    vocab = {}
    
    # getting the features
    vocab = get_word_freq([filename])
                    
    
    spam_val = 0
    ham_val = 0
    
    
    # finding the proabilities
    for word in vocab:
        if word in probabilities_by_category[0]:
            spam_val += vocab[word] * np.log(probabilities_by_category[0][word])

        if word in probabilities_by_category[1]:
            ham_val += vocab[word] * np.log(probabilities_by_category[1][word])   

            

    msg = "ham"
    
    if spam_val + np.log(prior_by_category[0]) > ham_val + np.log(prior_by_category[1]):
        msg = "spam"
        
    
    classify_result = (msg, [spam_val, ham_val])    
    
        
    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    
    alpha = [1E-30, 1E-15, 1E-5, 1E0, 1E2, 1E3]
    
    type1 = []
    type2 = []
    
    for a in alpha:
    
        performance_measures = np.zeros([2,2])
        
        # Classify emails from testing set and measure the performance
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                     probabilities_by_category,
                                                     priors_by_category)
            
            spam_val = log_posterior[0]
            ham_val = log_posterior[1]
            
            if spam_val + np.log(priors_by_category[0]) + np.log(a) > ham_val + np.log(priors_by_category[1]):
                label = "spam"
            else:
                label = "ham"
            
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1
            
        type1.append(performance_measures[0][1])
        type2.append(performance_measures[1][0])
        
    # plot 
    
    plt.rcParams["figure.figsize"] = (10,10)
    
    for i in range(0, len(type1)):
        plt.scatter(type1[i], type2[i])
        
    plt.xlabel("Type1 Error")
    plt.ylabel("Type2 Error")
    plt.title("Errors with different values of alpha")
    plt.legend(alpha, loc = "upper right")
    plt.grid()
    plt.show()
    
    
        
 