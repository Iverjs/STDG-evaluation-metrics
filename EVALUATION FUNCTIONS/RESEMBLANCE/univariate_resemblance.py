#import libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from math import sqrt
from scipy.spatial import distance
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

def basic_stats(df,df_name):
    """
    Calculate basic statistics (mean, standard deviation) for a DataFrame.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The DataFrame for which to calculate statistics.
    df_name : str
        The name to be used as the column header in the returned DataFrame
        
    Returns
    -------
    pandas.core.frame.DataFrame
        A DataFrame containing the mean and standard deviation for each column.
    """
    # get list of numerical column names
    num_col = (df.select_dtypes(include=['float64','int64'])).columns.tolist()
    stats = []

    # loop through columns
    for c in num_col:
        # drop NaN-values from column
        non_na_col = df[c].dropna()
        # calculate mean and std
        mean = non_na_col.mean()
        std = non_na_col.std()
        # add to lists
        stats.append(f'{mean:.2f} ± {std:.2f}')
        
    stats_df = pd.DataFrame({df_name:stats}, index=num_col)
    return stats_df
    
def student_t_tests(real, synthetic) :

    """Performs Student T-tests to compare numerical attributes of real data and synthetic data.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list the p-values of the statistical tests
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the p-values of the tests
    p_values = []

    #loop to perform the tests for each attribute
    for c in num_cols :
        # Drop NaN values from both real and synthetic data
        real_col = real[c].dropna()
        synth_col = synthetic[c].dropna()
        
        # perform T-test if enough datapoints in each variable
        if len(real_col) > 1 and len(synth_col) > 1:
            _, p = stats.ttest_ind(real_col, synth_col)
        else:
            p = float('nan')
            
        p_values.append(p)

    #return the obtained p-values
    return p_values


def mann_whitney_tests(real, synthetic) :

    """Performs Mann Whitney U-Tests to compare numerical attributes of real data and synthetic data.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list the p-values of the statistical tests
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the p-values of the tests
    p_values = []

    #loop to perform the tests for each attribute
    for c in num_cols :
        # Drop NaN values from both real and synthetic data
        real_col = real[c].dropna()
        synth_col = synthetic[c].dropna()
        
        # perform T-test if enough datapoints in each variable
        if len(real_col) > 1 and len(synth_col) > 1:
            _, p = stats.mannwhitneyu(real_col, synth_col)
        else:
            p = float('nan')
            
        p_values.append(p)

    #return the obtained p-values
    return p_values


def ks_tests(real, synthetic) :

    """Performs Kolmogorov Smirnov tests to compare numerical attributes of real data and synthetic data.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list the p-values of the statistical tests
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the p-values of the tests
    p_values = []

    #loop to perform the tests for each attribute
    for c in num_cols :
        # Drop NaN values from both real and synthetic data
        real_col = real[c].dropna()
        synth_col = synthetic[c].dropna()
        
        # perform T-test if enough datapoints in each variable
        if len(real_col) > 1 and len(synth_col) > 1:
            _, p = stats.ks_2samp(real_col, synth_col)
        else:
            p = float('nan')
            
        p_values.append(p)

    #return the obtained p-values
    return p_values


def chi_squared_tests(real, synthetic) :

    """Performs Chi-squared tests to compare categorical attributes of real data and synthetic data.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list the p-values of the statistical tests
    """

    #get list of categorical column names
    cat_cols = (real.select_dtypes(include=['category'])).columns

    #initialize a list to save the p-values of the tests
    p_values = []

    #loop to perform the tests for each attribute
    for c in cat_cols :
        # drop NaN values from data
        real_col = real[c].dropna()
        synth_col = synthetic[c].dropna()

        # ensure both columns are non-empty
        if len(real_col) > 1 and len(synth_col) > 1:
            #create contingency table
            observed = pd.crosstab(real_col, synth_col)
            #perform chi-squared test
            if observed.size>0:
                _, p, _, _ = chi2_contingency(observed)
            else:
                p = float('nan')
        else:
            p = float('nan')
        # add p-values to list    
        p_values.append(p)


def scale_data(df) :

    """Scale a dataframe to get the values between 0 and 1. It returns the scaled dataframe.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The dataframe to scale

    Returns
    -------
    pandas.core.frame.DataFrame
        A dataframe with the scaled data
    """
    # impute NaN values with the column mean
    df_filled = df.fillna(df.mean())
    
    #initialize and fit the scaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_filled)

    #return the scaled dataframe
    return pd.DataFrame(scaled, columns=df.columns.tolist())


def euclidean_distances(real, synthetic):
    """
    Compute the Euclidean distances between real data attributes and synthetic data attributes independently.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        A list with the distances values
    """

    # Get list of numerical column names
    num_cols = real.select_dtypes(include=['int64', 'float64']).columns

    # Initialize a list to save the distances values
    dists = []

    # Loop to perform the distances for each attribute
    for c in num_cols:
        dist = np.linalg.norm(real[c].values - synthetic[c].values)
        dists.append(dist)

    # return list of computed distances
    return dists


def cosine_distances(real, synthetic) :

    """Compute the cosine distances between real data attributes and synthetic data attributes independently.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list with the distances values
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the distances values
    dists = []

    #loop to perform the distances for each attribute
    for c in num_cols :
        dists.append(distance.cosine(real[c].values, synthetic[c].values))

    #return the list with the computed distances
    return dists


def js_distances(real, synthetic) :

    """Compute the Jenshen-Shannon distances between real data attributes and synthetic data attributes independently.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list with the distances values
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the distances values
    dists = []

    #loop to perform the distances for each attribute
    for c in num_cols :
        dists.append(distance.jensenshannon(stats.norm.pdf(real[c].values), stats.norm.pdf(synthetic[c].values)))

    #return the list with the computed distances
    return dists


def wass_distances(real, synthetic) :

    """Compute the Wasserstein distances between real data attributes and synthetic data attributes independently.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list with the distances values
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the distances values
    dists = []

    #loop to perform the distances for each attribute
    for c in num_cols :
        dists.append(stats.wasserstein_distance(real[c].values, synthetic[c].values))

    #return the list with the computed distances
    return dists
