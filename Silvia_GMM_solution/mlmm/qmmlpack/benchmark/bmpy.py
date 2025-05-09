#!/usr/bin/env python3

# qmmlpack
# (c) Matthias Rupp, 2006-2016.
# See LICENSE.txt for license.

"""Benchmarking driver.

Runs Python timings for code snippets.

Part of qmmlpack library.
"""

import os
import sys
import time
import datetime
import platform
import argparse
import importlib
import gc
import json
import math
import numpy as np
import scipy as sp
import scipy.stats

#  #######################
#  #  Utility functions  #
#  #######################

def bm_error(msg):
    """Prints error message and aborts."""
    print("Error:", msg)
    sys.exit(1)

def bm_format_time(time):
    """Formats time in s as days/hours/minutes/seconds."""
    t = round(time)  #  Drop sub-second part
    d = t // 86400; t = t - d*86400;  # days, 60*60*24s
    h = t //  3600; t = t - h* 3600;  # hours, 60*60s
    m = t //    60; t = t - m*   60;  # minutes, 60s
    s = t;
    return "{}d {:0>2d}:{:0>2d}:{:0>2d}".format(d, h, m, s)

bm_progress_start = None
bm_progress_last  = None

def bm_show_progress(theta, ind):
    """Prints progress indicator."""
    global bm_progress_start, bm_progress_last
    meter_size = 20; update_frequency = 3

    now = time.time()
    if bm_progress_start is None: bm_progress_start = now
    if bm_progress_last  is None: bm_progress_last  = 0
    if now - bm_progress_last < update_frequency: return
    bm_progress_last = now;

    output = bm_format_time(now - bm_progress_start)

    thetaflat = theta.reshape(-1, theta.shape[-1])
    pos = [i for (v,i) in zip(thetaflat,range(len(thetaflat))) if (v == theta[ind]).all()][0]  # there must be a better way to do this
    eta = (len(thetaflat) - (pos+1)) * len(thetaflat) / (pos+1)
    output = output + "  ETA " + bm_format_time(eta);

    percent = round(100 * (pos+1) / len(thetaflat))
    percent_meter = round(percent * meter_size / 100)
    output += "  " + "[".ljust(1+percent_meter, "*") + "]".rjust(1+meter_size-percent_meter, "-") + " " + str(percent) + "%";

    print(output)

def bm_repeated_timing(expr):
    """Evaluates expression repeatedly and returns average time in seconds, similar to Mathematica's RepeatedTiming[] function."""
    repetitions = 4  # expression time measurement is repeated at least this often
    min_time = 3 / repetitions  # minimum time in s to measure / per repetition

    gc.collect()
    gc.disable()

    times = np.zeros(repetitions)
    for i in range(repetitions):
        count = 0
        start = time.perf_counter()
        while time.perf_counter() - start < min_time:
            exec(expr, globals());
            count += 1
        times[i] = (time.perf_counter() - start) / count

    gc.enable()

    # condense timings into a single result
    # min is likely the better operator here, but we follow Mathematica's approach
    times = sp.stats.trim_mean(times, 0.05)

    return (times, bm_result)

#  #########################
#  #  Main body of script  #
#  #########################

#  No need for the "if __name__ == '__main__':" idiom here as this is a script anyway

#  ################################  #
#  #  Parse command line options  #  #
#  ################################  #

# default values of options
bm_force     = False
bm_progress  = False
bm_dry_run   = False

# parse command line arguments
parse = argparse.ArgumentParser(description="Benchmark timings for Python code snippets for qmmlpack development.", epilog="Part of qmmlpack library.")
parse.add_argument('-f', '--force', action='store_true', help='overwrites existing results file')
parse.add_argument('-p', '--progress', action='store_true', help='shows progress bar')
parse.add_argument('-n', '--dryrun', action='store_true', help='does not write results to file')
parse.add_argument('snippet', help='benchmark code snippet to be timed')
args = parse.parse_args()

bm_force, bm_progress, bm_dry_run, bm_snippet = args.force, args.progress, args.dryrun, args.snippet

# check command line arguments
if not os.path.isfile(bm_snippet): bm_error("snippet file '{}' not found.".format(bm_snippet))

#  ######################  #
#  #  Set up variables  #  #
#  ######################  #

# date and time the benchmark is run
bm_date_time_string = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M")

# machine on which benchmark is run
bm_system_descr = "{}, {}, Python {}".format(platform.node().split('.')[0], platform.platform(), platform.python_version())

# filename with actual benchmark code
bm_snippet = os.path.abspath(bm_snippet)

# directory where benchmarking driver is located. Assumed qmmlpack library subfolder
bm_basedir = os.path.abspath(os.path.dirname(sys.argv[0]))

# directory where result files are stored
bm_resultsdir = os.path.join(bm_basedir, "results")
if not os.path.isdir(bm_resultsdir): bm_error("could not access results directory '{}'.".format(bm_resultsdir))

# file in which results are stored
bm_results_file = os.path.splitext(os.path.basename(bm_snippet))[0] + "_" + bm_date_time_string + ".json"
bm_results_file = os.path.join(bm_resultsdir, bm_results_file)
if os.path.isfile(bm_results_file) and (not bm_force): bm_error("results file '{}' already exists.".format(bm_results_file))

#  ######################  #
#  #  Benchmarked code  #  #
#  ######################  #

# provide QMMLPack package
sys.path = [s for s in sys.path if not "qmmlpack" in s]
sys.path.append(os.path.abspath(os.path.join(bm_basedir, "..", "python")))

import qmmlpack as qmml

# provide convenience routines
def bm_outer(grida, gridb):
    """Outer product grid from two parameter vectors."""
    return np.asarray( [(a,b) for a in grida for b in gridb] ).reshape((len(grida), len(gridb), 2))

# execute benchmarked code snippets
with open(bm_snippet) as f:
    exec(f.read(), globals())

# process snippet definitions
if not 'function' in globals(): bm_error("snippet did not define function to be benchmarked.")

if not 'theta' in globals(): theta = None

if not 'theta_grid' in globals():
    if theta is None:
        theta_grid = None
    elif len(theta) == 1:
        theta_grid = np.asarray( [theta[0][1]] ).T
    elif len(theta) == 2:
        theta_grid = bm_outer(theta[0][1], theta[1][1])
    else:
        bm_error("only up to two parameters are currently supported.")

if not 'description' in globals(): bm_error("snippet did not define description of benchmark.")
if not isinstance(description, str): bm_error("snippet defined non-String description of benchmark.")

if not 'before' in globals():
    def before(*args):
        """Auto-defined benchmark set-up function, doing nothing."""
        pass

if not 'after' in globals():
    def after(*args):
        """Auto-defined benchmark tear-down function, doing nothing."""
        pass

#  #################  #
#  #  Run timings  #  #
#  #################  #

def bm_timing_function(thetaval=None, thetaind=None):
    """Calls benchmarked function, with or without parameter arguments."""
    if thetaval is None:
        # no-argument version
        before()
        (time, result) = bm_repeated_timing('bm_result = function()')
        after()
    else:
        # version with parameters
        before(thetaval, thetaind)
        (time, result) = bm_repeated_timing("bm_result = function({!r}, {!r})".format(list(thetaval), thetaind))
        after(thetaval, thetaind, result)
        if bm_progress: bm_show_progress(theta_grid, thetaind)
    return time

if theta is None:
    bm_timings = bm_timing_function()
else:
    # np.ndindex(theta.shape[:-1]) corresponds to levelspec {-2} in Mathematica
    bm_timings = np.asarray( [bm_timing_function(theta_grid[i], i) for i in np.ndindex(theta_grid.shape[:-1])] )
    bm_timings = bm_timings.reshape(theta_grid.shape[:-1])

#  ###########################  #
#  #  Export timing results  #  #
#  ###########################  #

bm_results = {
    "DateTime"  : bm_date_time_string,
    "Benchmark" : description,
    "System"    : bm_system_descr,
    "Theta"     : theta,
    "ThetaGrid" : theta_grid.tolist(),
    "Timings"   : bm_timings.tolist()
};

if not bm_dry_run:
    with open(bm_results_file, "w") as f:
        json.dump(bm_results, f, indent=4)
