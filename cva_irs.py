
# %matplotlib inline
from QuantLib import *
import numpy as Numpy
from math import *
import matplotlib.pyplot as Matplotlib

from grid import Grid
from helpers import CreateYieldTermStructure,CreateSwapTransaction,CalculateExposures,CreateDefaultTermStructures,Plot
import test_folder.test_func as ti

ti.testprint()

# initialize quantlib calendar  and first fixing 
calendar = TARGET()
startDate = Date(12, December, 2018)
settlementDate = Date(14, December, 2018)
endDate = Date(14, December, 2023)
gridStepPeriod = Period(1, Weeks)
firstIndexFixing = 0.0277594
Settings.instance().evaluationDate = startDate

# Helper class : Data structure for containing dates
# request simulation grid, define times and dates

grid = Grid(settlementDate, endDate, gridStepPeriod)
times = grid.GetTimes()
dates = grid.GetDates()
marketCurve = CreateYieldTermStructure()

# request swap transaction
# ==============================================================================
# This is creating the 3M Libor Index but dont know how
# What is relinkableYeildTermStrcutureHandle and look inside the USDLIBOR constructor 
# We might want to extend this to create other RFRs
# ==============================================================================
forecastingCurve = RelinkableYieldTermStructureHandle()
index = USDLibor(Period(3, Months), forecastingCurve)
transaction = CreateSwapTransaction(index)

# simulate exposures
nPaths = 25
exposures = CalculateExposures(marketCurve, forecastingCurve, index, firstIndexFixing, transaction, grid, nPaths)
Settings.instance().evaluationDate = startDate

# calculate expected positive and negative exposures
positiveExposures = exposures.copy()
positiveExposures[positiveExposures < 0.0] = 0.0
EPE = Numpy.mean(positiveExposures, axis = 0)

negativeExposures = exposures.copy()
negativeExposures[negativeExposures > 0.0] = 0.0
ENE = Numpy.mean(negativeExposures, axis = 0)

# request default term structures
recoveryRate = 0.4
DTS_ctpy, DTS_own = CreateDefaultTermStructures(startDate, recoveryRate, marketCurve)

# calculations for selected XVA metrics from simulated exposures
cvaTerms = 0.0
dvaTerms = 0.0
for i in range(grid.GetSteps()):
    df = marketCurve.discount(times[i + 1])
    dPD_ctpy = DTS_ctpy.defaultProbability(times[i + 1]) - DTS_ctpy.defaultProbability(times[i])
    dPD_own = DTS_own.defaultProbability(times[i + 1]) - DTS_own.defaultProbability(times[i])
    cvaTerms += df * 0.5 * (EPE[i + 1] + EPE[i]) * dPD_ctpy
    dvaTerms += df * 0.5 * (ENE[i + 1] + ENE[i]) * dPD_own

CVA = (1.0 - recoveryRate) * cvaTerms
DVA = (1.0 - recoveryRate) * dvaTerms

# print PV and XVA results
print('PV = ' + str(round(exposures[0][0], 0)))
print('Pre-margin CVA = ' + str(round(CVA, 0)))
print('Pre-margin DVA = ' + str(round(DVA, 0)))

# plot calculated exposures
# ==============================================================================
# TODO Shows only one plot at a time. Need to show all 
# ==============================================================================
Matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]
Plot(times, exposures, 'Exposures', 't')
Plot(times, positiveExposures, 'Positive exposures', 't')
Plot(times, negativeExposures, 'Negative exposures', 't')
Plot(times, EPE, 'Pre-margin EPE profile', 't')
Plot(times, ENE, 'Pre-margin ENE profile', 't')
Matplotlib.show()