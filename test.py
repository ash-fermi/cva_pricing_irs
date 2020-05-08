# from QuantLib import *

import QuantLib as ql
import pandas as pd

# ==============================================================================
# This can be used to create schedule 
# ==============================================================================
first_date = ql.Date(26, 10, 2018)
effective_date = ql.Date(26, 10, 2018)
termination_date = first_date + ql.Period(2,ql.Years)
tenor = ql.Period(ql.Quarterly)

calendar = ql.UnitedStates()
business_convention = ql.Following
termination_business_convention = ql.Following
date_generation = ql.DateGeneration.Forward
end_of_month = False

schedule = ql.Schedule(effective_date,
                       termination_date,
                       tenor,
                       calendar,
                       business_convention,
                       termination_business_convention,
                       date_generation,
                       end_of_month)

print(pd.DataFrame({'fixingdate': list(schedule),'accrualstartdate': list(schedule),'someRate': 0.03}))
# ==============================================================================
# Create foreign current bonds to represent the xccy swap legs
# ==============================================================================

# ==============================================================================
# Create all curves needed for discounting in diffrent CSAs 
# ==============================================================================