from Quantlib import *

calendar = TARGET()
start = Date(17,6,2019)
maturity = calendar.advance(start, Period('5y'))

fixedSchedule = MakeSchedule(start, maturity, Period('1Y'))


