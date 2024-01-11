##The purpose of this code is to determine the optimal time to purchase a home (if any)
##In order to minimize cumulative pre-retirement expenses toward the cost of housing

## Import Libraries
import numpy as np
import matplotlib.pyplot as plt

##Required user Inputs:
## 1a. Starting amount ($) in down-payment fund
#Note: If you are a first-time home-buyer, add government incentive to down (i.e., 5-10% times house price)
down = 400000
## 1b. Price of home ($) you wish to purchase 
house = 1500000
## 2. The constant monthly amount ($) invested into a down-payment account (if this varies month-to-month, a function should be provided instead)
down_additional = 5666
## 3. The expected monthly rent ($) paid in-place of mortgage
rent = 5000
## 4a. Mortgage Interest rate (decimal), assumed to be fixed for entire amortization
i = 0.054
## 4b. City's property tax rate (%) -- Assumes Toronto rate
city = 0.66
## 5. Amortization Period (in years)
n = 30
## 6. Years until you plan to retire (sell home date)
retire = 60
##7. Land Transfer Tax ($) (Approximate rate in Toronto)
LTT = house*0.016

## Define Mortgage Calculator 
def Mortgage(a,b,c,d):
    Mort = (a-b)*(((c/12)*(1+(c/12))**(12*d))/((1+(c/12))**(d*12)-1))
    return Mort
## Define Outstanding Loan Balance Calculator (amount you must pay back to bank if you do not fully pay off the home by retirement)
def OLB(a,b,c,d,e):
    Outstanding = (a-b)*(((1+(c/12))**(12*d)-(1+(c/12))**(e))/((1+(c/12))**(d*12)-1))
    return Outstanding


##This nested loop loops through every month from now until retiring month to calculate in how many months
##it would be optimal to purchase a home to minimize pre-retirement housing expenses
cumu = [] 
mortgage = []
for j in range(1,retire*12+1):
    down_update = []
    rent_update = []
    house_update = []
    olb = []
    months_amortized = []
    for k in range(1,retire*12+1):
        no_mortgage = False
        ##If prior to down payment month, continue paying rent
        if k < j:
            ##If down payment fund total is less than or equal to house, continue paying into down payment fund
            if (down + np.sum(np.array(down_update)) + down_additional) <= house:
                down_update.append(down_additional)
                rent_update.append(rent)
            ##If down payment fund will surpass house price following another down payment fund addition, stop paying into down fund
            else:
                down_update.append(0)
                rent_update.append(rent)
                no_mortgage = True
        ##If we have reached the down payment month (j), cease rent payments and pay down payment + first mortgage + property tax
        elif k == j:
            ##If you have already saved up for whole home, pay entire home price and skip mortgages
            if no_mortgage == True:
                down_final = house
                house_update.append(LTT + down_final + city*house/(12*100))
            else:
                down_final = np.sum(np.array(down_update))+down
                ##Update house fees with down payment, first month's mortgage, and the monthly property tax
                house_update.append(LTT + down_final + Mortgage(house,down_final,i,n) + city*house/(12*100))
                months_amortized.append(1)
        ##If we have surpassed down payment month (j), continue to pay mortgage (if applicable) and property tax
        elif k > j:
            ##If amortization period is still rolling, append mortgage and property tax
            if k-j < n*12-1:
                if no_mortgage == True:
                    house_update.append(city*house/(12*100))
                else:
                    down_final = np.sum(np.array(down_update))+down
                    house_update.append(Mortgage(house,down_final,i,n) + city*house/(12*100))
                    months_amortized.append(1)
            ##If amortization period is finished, only append property tax
            else:
                house_update.append(city*house/(12*100))
            
            ##Once we reach retirement, sell house and pay bank remaining fees if applicable.
            if k == retire*12:
                ##If you have already exceeded amortization period, no need to pay bank
                if k-j>=n*12 or no_mortgage == True:
                    olb.append(0)  
                else:
                    olb.append(OLB(house,down_final,i,n,np.sum(np.array(months_amortized))))

    ##If j == retirement, assume you pay rent entire lifespan
    if j/12 == retire:
        cumu.append(rent*12*retire)
        mortgage.append(0)
    else:
        Cumulative = np.sum(np.array(rent_update)) + np.sum(np.array(house_update)) + np.sum(np.array(olb)) - house
        mortgage.append(Mortgage(house,down_final,i,n))
        cumu.append(Cumulative)

plt.plot(np.array(range(len(np.array(cumu))))/12,np.array(cumu))
plt.scatter(np.argmin(np.array(cumu))/12,np.min(np.array(cumu)),color='red')
plt.ylabel("Cumulative ($) Lifetime Housing Expenditure")
plt.xlabel("Years (from now) Until Home Purchase")
plt.show() 

plt.plot(np.array(range(len(np.array(mortgage))))/12,np.array(mortgage))
plt.ylabel("Monthly Mortgage ($)")
plt.xlabel("Years (from now) Until Home Purchase")
plt.show() 

print('Optimal House Purchase is ', np.round(np.argmin(np.array(cumu))/12, 2), 'years from now')
print('Cumulative Housing Expense will be $', np.round(np.min(np.array(cumu)), 2))

        





    
