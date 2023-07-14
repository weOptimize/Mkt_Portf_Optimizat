#This file includes the function that returns the survival value for a given budgetting confidence policy.
#It is called from the main file
import math
import random
import numpy as np
import pandas as pd
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
from pandas_ods_reader import read_ods
from copulas.multivariate import GaussianMultivariate
from scipy.stats import rv_continuous, rv_histogram, norm, uniform, multivariate_normal, beta
from fitter import Fitter, get_common_distributions, get_distributions

#import created scripts:
from task_rnd_triang_with_interrupts_stdev_new_R2 import *
from task_rnd_triang_with_interrupts_stdev_new_R2_deterministic import *

#I define the number of candidates to be considered
initcandidates = 20
nr_confidence_policies = 1
mcs_leads = []
mcs_cost = []
maxbdgt = 3800
#initialize matrices to store leads and cost
leadsperproject_matrix = np.zeros((initcandidates, nr_confidence_policies))
costperproject_matrix = np.zeros((initcandidates, nr_confidence_policies))


#defining the function that calculates the total Leads of a portfolio of media
def portfolio_totalbudget(portfolio,leadsperproject):
    totalbudget_portfolio = 0
    #totalbudget_cost = 0
    for i in range(initcandidates):
        if portfolio[i] == 1:
            totalbudget_portfolio += leadsperproject[i]
            #totalbudget_cost += costperproject[i]
    #return totalbudget_portfolio, totalbudget_cost
    return totalbudget_portfolio


#define the function that returns the survival value for a given budgetting confidence policy
def survival_value_extractor(sim_leads, budgetting_confidence_policy, iterations):
    #calculate the cumulative sum of the values of the histogram
	valuesplus, base = np.histogram(sim_leads, bins=iterations) #it returns as many values as specified in bins valuesplus are frequencies, base the x-axis limits for the bins 
	cumulativeplus = np.cumsum(valuesplus)
	survivalvalues = 100*(len(sim_leads)-cumulativeplus)/len(sim_leads)
	#return index of item from survivalvalues that is closest to "1-budgetting_confidence_policy" typ.20%
	index = (np.abs(survivalvalues-100*(1-budgetting_confidence_policy))).argmin()
	#return value at base (which is indeed the durations that correspond to survival level) that matches the index
	budgetedduration = np.round(base[index],2)
	return budgetedduration
    

#define the function that returns the expected value for a given budgetting confidence policy
def expected_value_extractor(sim_cost, iterations):
    #calculate the cumulative sum of the values of the histogram
	valuesplus, base = np.histogram(sim_cost, bins=iterations) #it returns as many values as specified in bins valuesplus are frequencies, base the x-axis limits for the bins 
	cumulativeplus = np.cumsum(valuesplus)
	survivalvalues = 100*(len(sim_cost)-cumulativeplus)/len(sim_cost)
	#return index of item from survivalvalues that is closest to "1-budgetting_confidence_policy" typ.20%. Here I place 50% because I want to use the median=avg=E()
	index = (np.abs(survivalvalues-100*(1-.5))).argmin()
	#return value at base (which is indeed the durations that correspond to survival level) that matches the index
	budgetedcost = np.round(base[index],2)
	return budgetedcost



def simulate(arrayforsim, iterat):
    #initialize the arrays that will store the results of the MonteCarlo Simulation
    mcs_leads = []
    mcs_cost = []
    for i in range(len(arrayforsim)):        
        #if the value i is 1, then the simulation is performed
        if arrayforsim[i] == 1:
            # open ten different ODS files and store the results in a list after computing the CPM and MCS 
            # (restore to only last line if old version)
            if i < 9:
                filename = "RND_Schedules/data_wb0" + str(i+1) + ".ods"
            else:
                filename = "RND_Schedules/data_wb" + str(i+1) + ".ods"  
            #print(filename)
            mydata = read_ods(filename, 1)
            # open ten different ODS files and store the results in a list after computing the CPM and MCS 
            # (restore to only last line if old version)
            if i < 9:
                filename = "RND_Schedules/mediaplan_0" + str(i+1) + ".ods"
            else:
                filename = "RND_Schedules/mediaplan_" + str(i+1) + ".ods"
            #print(filename)
            myriskreg = read_ods(filename, 1) # was myriskreg = read_ods(filename, "Sheet1")

            #compute MonteCarlo Simulation and store the results in an array called "sim1_leads"
            sim_leads = MCS_CPM_RR(mydata, myriskreg, iterat)
            cashflows = []
            # open the file that contains the expected cash flows, and extract the ones for the project i (located in row i)
            with open('RND_Schedules/expected_cash_flows.txt') as f:
                # read all the lines in the file as a list
                lines = f.readlines()
                # get the line at index i (assuming i is already defined)
                line = lines[i]
                # split the line by whitespace and convert each element to a float
                cashflows = list(map(float, line.split()))

            # compute MonteCarlo Simulation and store the results in an array called "sim1_cost"
            #print(cashflows)
            sim_cost = MCS_cost(cashflows, iterat)
            #print(sim_cost)
            
            #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the cost at each iteration
            mcs_leads.append(sim_leads)
            mcs_cost.append(sim_cost)
            #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the cost at each iteration
            #mcs_costs1.append(sim1_cost)
            #compute the median of the cost results
            median_cost = expected_value_extractor(sim_cost, iterat)
        else:
            # if the value i is 0, then the simulation is not performed and "nothing is done" (was "the appended results an array full of zeros")
            # mcs_cost.append([0.0])   
            # mcs_leads.append(np.zeros(iterat))
            # do nothing and go to the next iteration
            pass
         
            

    # print ("mcs_leads", mcs_leads)
    # print ("mcs_cost", mcs_cost)
    return(mcs_leads, mcs_cost)

# compute the median of the cost results
def pointestimate(mcs_leads, mcs_cost, budgetting_confidence_policies, numberofprojects):
    #initialize the arrays that will store the point estimates with size nr of projects x nr of budgetting confidence policies
    leadsperproject_matrix = np.zeros((numberofprojects, len(budgetting_confidence_policies)))
    costperproject_matrix = np.zeros((numberofprojects, len(budgetting_confidence_policies)))
    for i in range(numberofprojects):
        median_cost = round(expected_value_extractor(mcs_cost[i], len(mcs_cost[i])),0)
        for j in range(len(budgetting_confidence_policies)):
            budgetting_confidence_policy = budgetting_confidence_policies[j]
            #extract the survival value from the array sim_duration that corresponds to the budgetting confidence policy
            survival_value = survival_value_extractor(mcs_leads[i], budgetting_confidence_policy, len(mcs_leads[i]))
            #store the first survival value in an array where the columns correspond to the budgetting confidence policies and the rows correspond to the projects
            leadsperproject_matrix[i][j]=survival_value
            costperproject_matrix[i][j]=median_cost/1000-survival_value #(was costperproject_matrix[i][j]=median_cost-survival_value and we must convert into thousand euros)
    # print ("leadsperproject_matrix", leadsperproject_matrix)
    # print ("costperproject_matrix", costperproject_matrix)
    return(leadsperproject_matrix, costperproject_matrix)

# modify MCS results to reflect the correlation matrix  
def correlatedMCS(mcs_results, iterat, nrcandidates, projection_indexes):  
    #check the parameters of beta distribution for each of the mcs_results  
    betaparams = []  
    for i in range(nrcandidates):  
        f = Fitter(mcs_results[0][i], distributions=['beta'])  
        f.fit()  
        betaparam=(f.fitted_param["beta"])  
        betaparams.append(betaparam)  
  
    #extract all "a" parameters from the betaparams array  
    a = []  
    for i in range(nrcandidates):  
        a.append(betaparams[i][0])  
  
    #extract all "b" parameters from the betaparams array  
    b = []  
    for i in range(nrcandidates):  
        b.append(betaparams[i][1])  
  
    #extract all "loc" parameters from the betaparams array  
    loc = []  
    for i in range(nrcandidates):  
        loc.append(betaparams[i][2])  
  
    #extract all "scale" parameters from the betaparams array  
    scale = []  
    for i in range(nrcandidates):  
        scale.append(betaparams[i][3])  
  
    # print("betaparams: ")
    # print(betaparams)  
  
    # copy the array with all MCS results  
    df0 = pd.DataFrame(data=mcs_results[0]).T  
    col_names = ["P{:02d}".format(i+1) for i in range(nrcandidates)]  
    df0.rename(columns=dict(enumerate(col_names)), inplace=True)  
    correlation_matrix0 = df0.corr()  
  
    # *********Correlation matrix with random values between 0 and 1, but positive semidefinite***************  
    # Set the seed value for the random number generator    
    seed_value = 1005    
    np.random.seed(seed_value)  
    # Generate a random symmetric matrix  
    A = np.random.rand(initcandidates, initcandidates)  
    A = (A + A.T) / 2  
    # Compute the eigenvalues and eigenvectors of the matrix  
    eigenvalues, eigenvectors = np.linalg.eigh(A)  
    # Ensure the eigenvalues are positive  
    eigenvalues = np.abs(eigenvalues)  
    # Normalize the eigenvalues so that their sum is equal to nrcandidates  
    eigenvalues = eigenvalues / eigenvalues.sum() * initcandidates  
    # Compute the covariance matrix. Forcing positive values, as long as negative correlations are not usual in reality of projects  
    cm10r = np.abs(eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T))  
    # Ensure the diagonals are equal to 1  
    for i in range(initcandidates):  
        cm10r[i, i] = 1  
    # print('cm10r BEFORE:')
   #  print(cm10r)
    
    # if the sum of the values inside projection_indexes is the same as the number of candidates, then we do not change the correlation matrix
    if len(projection_indexes) == initcandidates:
        cm10r = cm10r
    # if the sum of the values inside projection_indexes is not the same as the number of candidates, then we change the correlation matrix
    else:
        # we change the correlation matrix by setting the correlation between the candidates that are not selected to 0
        # for each time we find a zero, we trim the whole column adn row of the correlation matrix corresponding to that candidate
        i = 0
        j = 0
        for i in range(initcandidates):
            if i not in projection_indexes:
                cm10r = np.delete(cm10r, i-j, 0)
                cm10r = np.delete(cm10r, i-j, 1)
                j+=1
    # print('cm10r AFTER:')
    # print(cm10r)

    #make sure no legend appears in the next plot
    plt.figure(12)
    #plt.legend().set_visible(False)
    #heatmap of the correlation matrix cm10r
    sns.set(font_scale=1.15)
    sns.heatmap(cm10r, annot=True, cmap="Greys")

    #initialize dataframe df10r with size nrcandidates x iterations  
    df10r = pd.DataFrame(np.zeros((iterat, nrcandidates)))  
    # step 1: draw random variates from a multivariate normal distribution   
    # with the targeted correlation structure  
    r0 = [0] * cm10r.shape[0]                       # create vector r with as many zeros as correlation matrix has variables (row or columns)  
    mv_norm = multivariate_normal(mean=r0, cov=cm10r)    # means = vector of zeros; cov = targeted corr matrix  
    rand_Nmv = mv_norm.rvs(iterat)                               # draw N random variates  
    # step 2: convert the r * N multivariate variates to scores   
    rand_U = norm.cdf(rand_Nmv)   # use its cdf to generate N scores (probabilities between 0 and 1) from the multinormal random variates  
    # step 3: instantiate the nrcandidates marginal distributions   
    d_list = []  
    for i in range(nrcandidates):  
        d = beta(a[i], b[i], loc[i], scale[i])  
        d_list.append(d)  
    # draw N random variates for each of the nrcandidates marginal distributions  
    # WITHOUT applying a copula
    # do it only for the ones different from 0  
    rand_list = [d.rvs(iterat) for d in d_list]  
    # rand_list = [d.rvs(iterat) for i, d in enumerate(d_list) if i not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]  
    # initial correlation structure before applying a copula  
    c_before = np.corrcoef(rand_list)  
    # step 4: draw N random variates for each of the nrcandidates marginal distributions  
    # and use as inputs the correlated uniform scores we have generated in step 2  
    rand_list = [d.ppf(rand_U[:, i]) for i, d in enumerate(d_list)]  
    # final correlation structure after applying a copula  
    c_after = np.corrcoef(rand_list)  
    #print("Correlation matrix before applying a copula:")  
    #print(c_before)  
    #print("Correlation matrix after applying a copula:")  
    #print(c_after)  
    # step 5: store the N random variates in the dataframe  
    for i in range(nrcandidates):  
        df10r[i] = rand_list[i]  
    col_names = ["P{:02d}".format(i+1) for i in range(nrcandidates)]  
    df10r.rename(columns=dict(enumerate(col_names)), inplace=True)  
    correlation_matrix1 = df10r.corr()  
    return df10r  

def calc_det(arrayforsim, iterat):
    #initialize the arrays that will store the results of the MonteCarlo Simulation
    det_leads = []
    det_cost = []
    for i in range(len(arrayforsim)):        
        #if the value i is 1, then the simulation is performed
        if arrayforsim[i] == 1:
            # open ten different ODS files and store the results in a list after computing the CPM and MCS 
            # (restore to only last line if old version)
            if i < 9:
                filename = "RND_Schedules/data_wb0" + str(i+1) + ".ods"
            else:
                filename = "RND_Schedules/data_wb" + str(i+1) + ".ods"  
            #print(filename)
            mydata = read_ods(filename, 1)
            # open ten different ODS files and store the results in a list after computing the CPM and MCS 
            # (restore to only last line if old version)
            if i < 9:
                filename = "RND_Schedules/riskreg_0" + str(i+1) + ".ods"
            else:
                filename = "RND_Schedules/riskreg_" + str(i+1) + ".ods"
            #print(filename)
            myriskreg = read_ods(filename, 1) # was myriskreg = read_ods(filename, "Sheet1")

            #compute MonteCarlo Simulation and store the results in an array called "sim1_leads"
            sim_leads = MCS_CPM_RRdet(mydata, myriskreg, iterat)
            cashflows = []
            # open the file that contains the expected cash flows, and extract the ones for the project i (located in row i)
            with open('RND_Schedules/expected_cash_flows.txt') as f:
                # read all the lines in the file as a list
                lines = f.readlines()
                # get the line at index i (assuming i is already defined)
                line = lines[i]
                # split the line by whitespace and convert each element to a float
                cashflows = list(map(float, line.split()))

            # compute MonteCarlo Simulation and store the results in an array called "sim1_cost", also 
            sim_cost = MCS_COSTdet(cashflows, iterat)
            print(sim_cost)
            # substract sim_leads from all the values inside the array
            for j in range(len(sim_cost)):
                sim_cost[j] = sim_cost[j] - sim_leads[j]
            #print(sim_cost)
            
            #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the cost at each iteration
            det_leads.append(sim_leads)
            det_cost.append(sim_cost)
            #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the cost at each iteration
            #mcs_costs1.append(sim1_cost)
            #compute the median of the cost results
        else:
            # if the value i is 0, then the simulation is not performed and "nothing is done" (was "the appended results an array full of zeros")
            # mcs_cost.append([0.0])   
            # mcs_leads.append(np.zeros(iterat))
            # do nothing and go to the next iteration
            pass

    return(det_leads, det_cost)
