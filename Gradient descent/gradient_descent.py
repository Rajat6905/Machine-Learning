import math 
import pandas as pd
import matplotlib.pyplot as plt



def gradient_descent(x,y):
    # set m and b to 0
    m_curr = 0
    b_curr = 0
    iterations = 1000000
    n = len(x)
    
    learning_rate = 0.0002

    cost_previous = 0
    plt.scatter(x,y,color='red',marker='+',linewidth='5')
    for i in range(iterations):
        
        #for linear regression we have equation
        #y=mx+b
        y_predicted = m_curr * x + b_curr
        
        cost = (1/n)*sum([value**2 for value in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))   #m delivative
        bd = -(2/n)*sum(y-y_predicted)       #b delivative
        #Update m and b
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            plt.plot(x,y_predicted,color='green')
            break
        cost_previous = cost
#        print ("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost, i))

    return m_curr, b_curr

if __name__ == "__main__":

    dataset=pd.read_csv("student_scores.csv")
    
    x=dataset.iloc[:,0].values
    y=dataset.iloc[:,1].values

    m, b = gradient_descent(x,y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))
