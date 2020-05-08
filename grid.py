# TODO : Remove import * with proper imports
from QuantLib import *
# ==============================================================================
# # class for hosting simulation grid (dates, times)
# ==============================================================================
class Grid:
    def __init__(self, startDate, endDate, tenor):
        # create date schedule, ignore conventions and calendars
        self.schedule = Schedule(startDate, endDate, tenor, NullCalendar(), 
            Unadjusted, Unadjusted, DateGeneration.Forward, False)
        self.dayCounter = Actual365Fixed()
        self.tenor = tenor
    def GetDates(self):
        # get list of scheduled dates
        dates = []
        [dates.append(self.schedule[i]) for i in range(self.GetSize())]
        return dates            
    def GetTimes(self):
        # get list of scheduled times
        times = []
        [times.append(self.dayCounter.yearFraction(self.schedule[0], self.schedule[i])) 
            for i in range(self.GetSize())]
        return times
    def GetMaturity(self):
        # get maturity in time units
        return self.dayCounter.yearFraction(self.schedule[0], self.schedule[self.GetSteps()])
    def GetSteps(self):
        # get number of steps in schedule
        return self.GetSize() - 1    
    def GetSize(self):
        # get total number of items in schedule
        return len(self.schedule)    
    def GetTimeGrid(self):
        # get QuantLib TimeGrid object, constructed by using list of scheduled times
        return TimeGrid(self.GetTimes(), self.GetSize())
    def GetDt(self):
        # get constant time step
        return self.GetMaturity() / self.GetSteps()
    def GetTenor(self):
        # get grid tenor
        return self.tenor
