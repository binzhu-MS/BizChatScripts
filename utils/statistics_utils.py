"""
Statistical utility functions for metrics analysis.

This module provides statistical functions for analyzing experimental data,
including t-tests and proportion tests.

Author: Bin Zhu
Date: July 31, 2025
"""

import logging
import statistics as builtin_statistics

from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp


logger = logging.getLogger(__name__)

DIFF_COLUMNS = ['control', 'experiment', 'diff', 'prop_diff', 'pval']


def tsingle(values, mean=0, alternative="two-sided"):
    """Perform one-sample t-test."""
    result = ttest_1samp(values, mean, alternative=alternative)
    stat = safe_round(result[0], 3)  # statistic
    pval = safe_round(result[1], 3)  # p-value
    return {'stat': stat, 'pval': pval}


def format_stat(test_result):
    """Format statistical test result for display."""
    return f"{test_result['stat']} (p={test_result['pval']})"


def tdiff(control, experiment, paired=True):
    """
    Perform t-test comparison between control and experiment groups.
    
    Args:
        control: List of control group values
        experiment: List of experiment group values  
        paired: If True, perform paired t-test; if False, independent t-test
        
    Returns:
        dict: Statistical results including means, differences, and p-value
    """
    control = [v for v in control if v is not None]
    experiment = [v for v in experiment if v is not None]
    
    try:
        control_mean = safe_round(builtin_statistics.mean(control), 3)
    except:
        control_mean = None
    try:
        experiment_mean = safe_round(builtin_statistics.mean(experiment), 3)
    except:
        experiment_mean = None
        
    if control_mean is not None and experiment_mean is not None:
        diff = safe_round(experiment_mean - control_mean, 3)
        prop_diff = safe_round(safe_prop(diff, control_mean), 3)
    else:
        diff = None
        prop_diff = None
        
    try:
        if paired:
            result = ttest_rel(experiment, control)
        else:
            result = ttest_ind(experiment, control, equal_var=False)
        
        stat = safe_round(result[0], 3)  # statistic
        pval = safe_round(result[1], 3)  # p-value
    except:
        stat = None
        pval = None
        
    return {
        'control': control_mean, 
        'experiment': experiment_mean, 
        'diff': diff, 
        'prop_diff': prop_diff, 
        'stat': stat, 
        'pval': pval
    }


def bdiff(control_k, experiment_k, control_n, experiment_n):
    """Perform proportion test for binary outcomes."""
    control_p = safe_prop(control_k, control_n)
    experiment_p = safe_prop(experiment_k, experiment_n)
    diff = safe_diff(experiment_p, control_p)
    prop_diff = safe_prop(diff, control_p)
    zstat, pval = proportions_ztest([control_k, experiment_k], [control_n, experiment_n])
    return {
        'control': safe_round(control_p, 3), 
        'experiment': safe_round(experiment_p, 3), 
        'diff': safe_round(diff, 3), 
        'prop_diff': safe_round(prop_diff, 3), 
        'stat': safe_round(zstat, 3), 
        'pval': safe_round(pval, 3)
    }


def rawdiff(control, experiment):
    """Calculate raw difference without statistical testing."""
    diff = safe_round(safe_diff(experiment, control), 3)
    prop_diff = safe_round(safe_prop(diff, control), 3)
    return {
        'control': safe_round(control, 3), 
        'experiment': safe_round(experiment, 3), 
        'diff': diff, 
        'prop_diff': prop_diff, 
        'stat': None, 
        'pval': None
    }


def safe_prop(k, n):
    """Safely calculate proportion handling edge cases."""
    if n is None or k is None:
        return None
    elif n == 0:
        if k == 0:
            return 0
        elif k > 0:
            return "Inf"
        else:
            return "-Inf"
    else:
        if k is None:
            k = 0
        return k / n


def safe_diff(a, b):
    """Safely calculate difference handling None values."""
    try:
        if a == b:
            return 0
        else:
            return a - b
    except TypeError:
        return None


def safe_round(v, digits=0):
    """Safely round values handling None and non-numeric types."""
    try:
        return round(v, digits)
    except:
        return v


def safe_mean(values):
    """Safely calculate mean handling None values."""
    clean_values = [v for v in values if v is not None]
    if len(clean_values) == 0:
        return None
    else:
        return builtin_statistics.mean(clean_values)
