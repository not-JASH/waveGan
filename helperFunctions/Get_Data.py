from binance.client import Client
from datetime import timedelta
from numpy import savetxt


def num2date(dateNum,isLeap,year):
    if dateNum < 32 :
        month = "January"
        day = dateNum
    elif dateNum < 60 + isLeap:
        month = "February"
        day = dateNum - 31
    elif dateNum < 91 + isLeap:
        month = "March"
        day = dateNum - (59+isLeap)
    elif dateNum < 121 + isLeap:
        month = "April"
        day = dateNum - (90+isLeap)
    elif dateNum < 152 + isLeap:
        month = "May"
        day = dateNum - (120+isLeap)
    elif dateNum < 182 + isLeap:
        month = "June"
        day = dateNum - (151+isLeap)
    elif dateNum < 213 + isLeap:
        month = "July"
        day = dateNum - (181+isLeap)
    elif dateNum < 244 + isLeap:
        month = "August"
        day = dateNum - (212+isLeap)
    elif dateNum < 274 + isLeap:
        month = "September"
        day = dateNum - (243+isLeap)
    elif dateNum < 305 + isLeap:
        month = "October"
        day = dateNum - (273+isLeap)
    elif dateNum < 335 + isLeap:
        month = "November"
        day = dateNum - (304+isLeap)
    else:
        month = "December"
        day = dateNum - (334+isLeap)

    day = str(day);
    year = str(year);
    datestring = day+","+month+","+year
    return datestring
#############
#IMPORTANT put your api key here
api_key     = "binance api key"
api_secret  = "binance api secret"

client=Client(api_key,api_secret)

for i in range(104):
    ll_day = 7*i + 1
    ul_day = ll_day + 6

    ll_year = 2019
    ll_isleap = 0
    ul_year = 2019
    ul_isleap = 0

    if ll_day > 365:
        ll_year = 2020
        ll_isleap = 1
        ll_day -= 365
    if ul_day > 365:
        ul_year = 2020
        ul_isleap = 1
        ul_day -= 365

    dayone = num2date(ll_day,ll_isleap,ll_year)
    daytwo = num2date(ul_day,ul_isleap,ul_year)
    print(dayone+"\t"+daytwo+"\n")
    klines=client.get_historical_klines("BTCUSDT",client.KLINE_INTERVAL_1MINUTE,dayone,daytwo)
    textname = "BTCUSDT_"+str(i+1)+".txt"
    savetxt("../binanceData/"+textname,klines,delimiter=" ",fmt="%s")

