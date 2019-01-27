import pandas as pd
data=pd.read_csv("diabetes.csv")


labels = ['low','medium','high']
for j in data.columns[:-1]:
    mean = data[j].mean()
    data[j] = data[j].replace(0,mean)
    data[j] = pd.cut(data[j],bins=len(labels),labels=labels)

def count(data,colname,label,target):
    condition = (data[colname] == label) & (data['Outcome'] == target)
    return len(data[condition])
predicted = []
probabilities = {0:{},1:{}}

train_percent = 70
train_len = int((train_percent*len(data))/100)
train_X = data.iloc[:train_len,:]
test_X = data.iloc[train_len+1:,:-1]
test_y = data.iloc[train_len+1:,-1]

count_0 = count(train_X,'Outcome',0,0)
count_1 = count(train_X,'Outcome',1,1)
    
prob_0 = count_0/len(train_X)
prob_1 = count_1/len(train_X)

for col in train_X.columns[:-1]:
        probabilities[0][col] = {}
        probabilities[1][col] = {}
        
        for category in labels:
            count_ct_0 = count(train_X,col,category,0)
            count_ct_1 = count(train_X,col,category,1)
            
            probabilities[0][col][category] = count_ct_0 / count_0
            probabilities[1][col][category] = count_ct_1 / count_1
            
            
for row in range(0,len(test_X)):
        prod_0 = prob_0
        prod_1 = prob_1
        for feature in test_X.columns:
            prod_0 *= probabilities[0][feature][test_X[feature].iloc[row]]
            prod_1 *= probabilities[1][feature][test_X[feature].iloc[row]]
        
        if prod_0 > prod_1:
            predicted.append(0)
        else:
            predicted.append(1)

from sklearn import metrics 
metrics.confusion_matrix(predicted,test_y)



#X=data.iloc[:,:-1]
#y=data.iloc[:,-1]
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)
#
#from sklearn.naive_bayes import GaussianNB
#clf=GaussianNB()
#clf.fit(X_train,y_train)
#y_pred=clf.predict(X_test)
#from sklearn import metrics
#print("Classification Report",metrics.classification_report(y_pred,y_test))
#print("Confusion Metrix:\n",metrics.confusion_matrix(y_pred,y_test))
#print("Accurecy Score:",metrics.accuracy_score(y_pred,y_test))

