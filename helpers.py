# TODO : Remove import * with proper imports

from QuantLib import *
import numpy as np
from math import exp
import matplotlib.pyplot as Matplotlib
    

# create hard-coded yield term structure
# TODO : Remove this hardcoding with pullig the curve from an excel sheet or REST API

def CreateYieldTermStructure():
    termStructureDates = [Date(12, December, 2018), Date(13, December, 2018), Date(19, December, 2018), \
        Date(12, January, 2019), Date(12, February, 2019), Date(12, March, 2019), Date(12, June, 2019), \
        Date(12, December, 2019), Date(12, December, 2020), Date(12, December, 2021), Date(12, December, 2022), \
        Date(12, December, 2023), Date(12, December, 2024), Date(12, December, 2025), Date(12, December, 2026), \
        Date(12, December, 2027), Date(12, December, 2028), Date(12, December, 2029), Date(12, December, 2030), \
        Date(12, December, 2033), Date(12, December, 2038), Date(12, December, 2043), Date(12, December, 2048), \
        Date(12, December, 2058), Date(12, December, 2068)]
    
    # USD Libor zero swap discount factors from swap manager as of 12.12.2018
    termStructureDiscountFactors = [1.0, 0.999758, 0.99957, 0.99792, 0.995597, 0.993108, 0.985659, 0.969519, \
        0.94423, 0.918219, 0.892931, 0.86814, 0.843459, 0.819195, 0.794985, 0.770981, 0.747243, 0.724021, \
        0.701601, 0.639109, 0.548016, 0.472681, 0.408786, 0.310937, 0.239833]
    
    # create yield term structure from a given set of discount factors
    yieldTermStructure = DiscountCurve(termStructureDates, termStructureDiscountFactors, Actual360(), TARGET())
    yieldTermStructure.enableExtrapolation()
    return yieldTermStructure

# create hard-coded swap transaction
# TODO : Add this to another class called instruments and add more products
def CreateSwapTransaction(index):
    # create benchmarking IR receiver swap, PV(t = 0) = 0.0
    fixedSchedule = Schedule(Date(14, December, 2018), Date(14, December, 2023), Period(1, Years), TARGET(), \
        ModifiedFollowing, ModifiedFollowing, DateGeneration.Backward, False)

    floatingSchedule = Schedule(Date(14, December, 2018), Date(14, December, 2023), Period(3, Months), TARGET(), \
        ModifiedFollowing, ModifiedFollowing, DateGeneration.Backward, False)

    swap = VanillaSwap(VanillaSwap.Receiver, 10000000.0, fixedSchedule, 0.03, Actual365Fixed(), \
        floatingSchedule, index, 0.001277206920730623, Actual360())
    return swap


# term A(t, T) for analytical Hull-White zero-coupon bond price
def A(curve, a, sigma, t, T):
    f = curve.forwardRate(t, t, Continuous, NoFrequency).rate()
    value = B(a, t, T) * f - 0.25 * sigma * B(a, t, T) * sigma * B(a, t, T) * B(a, 0.0, 2.0 * t)
    return exp(value) * curve.discount(T) / curve.discount(t)


# term B(t, T) for analytical Hull-White zero-coupon bond price
def B(a, t, T):
    return (1.0 - exp(-a * (T - t))) / a
 

# workhorse: simulate future exposures for a given transaction using HW1F process
# TODO: Optimize this and also make it pluggable for other instruments 

def CalculateExposures(rateTermStructure, forecastingCurve, index, latestIndexFixing, transaction, grid, nPaths):
    # create vector of dates and times from the grid
    times = np.array(grid.GetTimes())
    dates = np.array(grid.GetDates())
    
    # request Hull-White 1-factor process and calibrated model parameters
    # generate paths for short rate
    process, a, sigma = CreateStochasticProcess(rateTermStructure)
    paths = GeneratePaths(process, grid.GetTimeGrid(), nPaths)
    
    # request transaction floating leg fixings dates
    scheduleDates = np.array(list(transaction.floatingSchedule()))
    transactionFixingDates = np.array([index.fixingDate(scheduleDates[i]) for i in range(scheduleDates.shape[0])])
    transactionFixingRates = np.zeros(shape = (transactionFixingDates.shape[0]))
    
    # create fixing dates, request simulated fixings
    # correction for the first observed fixing
    simulatedFixingDates = np.array(dates) - Period(index.fixingDays(), Days)
    simulatedFixingRates = np.mean(paths[:], axis = 0)
    simulatedFixingRates[0] = latestIndexFixing
    
    # create fixings lookup table
    fixingsLookupTable = {}
    for i in range(simulatedFixingDates.shape[0]):
        fixingsLookupTable[simulatedFixingDates[i]] = simulatedFixingRates[i]
    
    # add transaction fixing rates for a given date from fixings lookup table
    for i in range(transactionFixingDates.shape[0]):
        if transactionFixingDates[i] in fixingsLookupTable:
            transactionFixingRates[i] = fixingsLookupTable[transactionFixingDates[i]]
        else:
            # find the nearest fixing from lookup table
            transactionFixingRates[i] = \
            fixingsLookupTable.get(transactionFixingDates[i], \
            fixingsLookupTable[min(fixingsLookupTable.keys(), \
            key = lambda d: abs(d - transactionFixingDates[i]))])
    
    # add required transaction fixing dates and rates to floating leg index
    index.addFixings(transactionFixingDates, transactionFixingRates, True)

    # create containers for simulated curves and exposures
    curves = np.zeros(shape = (nPaths, grid.GetSize()), dtype = DiscountCurve)
    exposures = np.zeros(shape = (nPaths, grid.GetSize()), dtype = float)
    # set the first curve to be current spot market curve
    curves[:, 0] = rateTermStructure
    
    # loop through time steps
    for t in range(1, grid.GetSteps()):
        # loop through all paths        
        for s in range(nPaths):
            # set a new starting date and create a new list of dates
            curveDate = dates[t]
            gridTenor = grid.GetTenor()
            curveDates = [curveDate] + [curveDate + (gridTenor * k) for k in range(1, grid.GetSize())]
            rt = paths[s, t]
            # define list for simulated zero-coupon bonds
            # set the first discount factor to be 1.0
            zeros = np.zeros(shape = (grid.GetSize()), dtype = float)
            zeros[0] = 1.0
            dt = grid.GetDt()
            for k in range(1, grid.GetSize()):
                # use analytic formula for Hull-White zero-coupon bond price
                A_term = A(rateTermStructure, a, sigma, times[t], times[t] + (dt * k))
                B_term = B(a, times[t], times[t] + (dt * k))
                zeros[k] = A_term * exp(-B_term * rt)
            
            # create a new curve from simulated zero-coupon bond prices
            curves[s][t] = DiscountCurve(curveDates, zeros, Actual360(), TARGET())
            curves[s][t].enableExtrapolation()

    # link transaction and pricing engine
    # TODO : This is for a IRS we need FX swap and Forwards and XCCY and Basis swaps 
    discountingCurve = RelinkableYieldTermStructureHandle()
    swapEngine = DiscountingSwapEngine(discountingCurve)
    transaction.setPricingEngine(swapEngine)
    
    # loop through grid time steps
    for t in range(grid.GetSteps()):
        # move forward in time along the grid
        Settings.instance().evaluationDate = dates[t]
        # calculate pv for transaction
        # link discounting and forecasting curves to a simulated curve
        for s in range(nPaths):
            curve = curves[s][t]
            discountingCurve.linkTo(curve)
            forecastingCurve.linkTo(curve)
            # save pv to exposure matrix
            exposures[s][t] = transaction.NPV()            
            # need a way to feed in FX and Cross Currency Swao MTM here
    return exposures

# create HW1F stochastic process using parameters calibrated to hard-coded set of flat swaption vols 
def CreateStochasticProcess(rateTermStructure):
    # hard-coded flat swaption volatilities (1x5, 2x4, 3x3, 4x2, 5x1)
    volatility = [0.2, 0.2, 0.2, 0.2, 0.2]
    endCriteria = EndCriteria(10000, 100, 0.000001, 0.00000001, 0.00000001)
    calibrator = ModelCalibrator(endCriteria)

    # add swaption helpers to calibrator
    for i in range(len(volatility)):
        t = i + 1; tenor = len(volatility) - i    
        helper = SwaptionHelper(
            Period(t, Years), 
            Period(tenor, Years), 
            QuoteHandle(SimpleQuote(volatility[i])), 
            USDLibor(Period(3, Months), YieldTermStructureHandle(rateTermStructure)), 
            Period(1, Years), 
            Actual360(), 
            Actual360(), 
            YieldTermStructureHandle(rateTermStructure))    
        calibrator.AddCalibrationHelper(helper)

    # create and calibrate model parameters
    model = HullWhite(YieldTermStructureHandle(rateTermStructure))
    engine = JamshidianSwaptionEngine(model)
    fixedParameters = []
    calibrator.Calibrate(model, engine, YieldTermStructureHandle(rateTermStructure), fixedParameters)

    # request calibrated parameters, create stochastic process object
    reversionSpeed = model.params()[0]
    rateVolatility = model.params()[1]
    stochasticProcess = HullWhiteProcess(YieldTermStructureHandle(rateTermStructure), reversionSpeed, rateVolatility)    
    return stochasticProcess, reversionSpeed, rateVolatility



# ==============================================================================
# # class for hosting calibration helpers and calibration procedure for a given model
# ==============================================================================

class ModelCalibrator:
    def __init__(self, endCriteria):        
        self.endCriteria = endCriteria
        self.helpers = []
    def AddCalibrationHelper(self, helper):
        self.helpers.append(helper)
    def Calibrate(self, model, engine, curve, fixedParameters):
        # assign pricing engine to all calibration helpers
        for i in range(len(self.helpers)):
            self.helpers[i].setPricingEngine(engine)
        method = LevenbergMarquardt()
        if(len(fixedParameters) == 0):
            model.calibrate(self.helpers, method, self.endCriteria)
        else:
            model.calibrate(self.helpers, method, self.endCriteria,
                NoConstraint(), [], fixedParameters)

# path generator method for uncorrelated and correlated 1-D stochastic processes
def GeneratePaths(process, timeGrid, n):
    # for correlated processes, use GaussianMultiPathGenerator
    if(type(process) == StochasticProcessArray):
        times = []; [times.append(timeGrid[t]) for t in range(len(timeGrid))]        
        nGridSteps = (len(times) - 1) * process.size()
        sequenceGenerator = UniformRandomSequenceGenerator(nGridSteps, UniformRandomGenerator())
        gaussianSequenceGenerator = GaussianRandomSequenceGenerator(sequenceGenerator)
        pathGenerator = GaussianMultiPathGenerator(process, times, gaussianSequenceGenerator, False)        
        paths = np.zeros(shape = (n, process.size(), len(timeGrid)))
        
        # loop through number of paths
        for i in range(n):
            # request multiPath, which contains the list of paths for each process
            multiPath = pathGenerator.next().value()
            # loop through number of processes
            for j in range(multiPath.assetNumber()):
                # request path, which contains the list of simulated prices for a process
                path = multiPath[j]
                # push prices to array
                paths[i, j, :] = np.array([path[k] for k in range(len(path))])
        # resulting array dimension: n, process.size(), len(timeGrid)
        return paths

    # for uncorrelated processes, use GaussianPathGenerator
    else:
        sequenceGenerator = UniformRandomSequenceGenerator(len(timeGrid), UniformRandomGenerator())
        gaussianSequenceGenerator = GaussianRandomSequenceGenerator(sequenceGenerator)
        maturity = timeGrid[len(timeGrid) - 1]
        pathGenerator = GaussianPathGenerator(process, maturity, len(timeGrid), gaussianSequenceGenerator, False)
        paths = np.zeros(shape = (n, len(timeGrid)))
        for i in range(n):
            path = pathGenerator.next().value()
            paths[i, :] = np.array([path[j] for j in range(len(timeGrid))])
        # resulting array dimension: n, len(timeGrid)
        return paths

# create hard-coded default term structures
# TODO 1: Remove hard coding and get input from excel 
# TODO 2: enhance this to accept portfolio of transactions with different counterparties

def CreateDefaultTermStructures(startDate, recoveryRate, rateTermStructure):
    # hard-coded flat CDS term structures for counterparty and institution
    CDS_tenors = [Period(6, Months), Period(1, Years), Period(2, Years), Period(3, Years), \
        Period(4, Years), Period(5, Years), Period(7, Years), Period(10, Years)]
    CDS_ctpy = [100, 100, 100, 100, 100, 100, 100, 100]
    CDS_self = [100, 100, 100, 100, 100, 100, 100, 100]

    # create CDS helpers for counterparty
    CDSHelpers_ctpy = [SpreadCdsHelper((CDS_spread / 10000.0), CDS_tenor, 0, TARGET(), Quarterly, Following, \
        DateGeneration.TwentiethIMM, Actual365Fixed(), recoveryRate, YieldTermStructureHandle(rateTermStructure))
    for CDS_spread, CDS_tenor in zip(CDS_ctpy, CDS_tenors)] 

    # create CDS helpers for self
    CDSHelpers_self = [SpreadCdsHelper((CDS_spread / 10000.0), CDS_tenor, 0, TARGET(), Quarterly, Following, \
        DateGeneration.TwentiethIMM, Actual365Fixed(), recoveryRate, YieldTermStructureHandle(rateTermStructure))
    for CDS_spread, CDS_tenor in zip(CDS_self, CDS_tenors)]

    # create default term structures
    defaultCurve_ctpy = PiecewiseFlatHazardRate(startDate, CDSHelpers_ctpy, Actual365Fixed()) 
    defaultCurve_self = PiecewiseFlatHazardRate(startDate, CDSHelpers_self, Actual365Fixed()) 
    return defaultCurve_ctpy, defaultCurve_self

# wrapper for matplotlib
def Plot(x, y, title, xlabel):
    if(y.ndim == 1):
        Matplotlib.plot(x, y)
    else:
        for i in range(y.shape[0]):
            y_i = y[i, :] 
            Matplotlib.plot(x, y_i)
    Matplotlib.title(title)
    Matplotlib.xlabel(xlabel)
    Matplotlib.show()