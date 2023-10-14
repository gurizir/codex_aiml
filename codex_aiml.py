#step 1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#step 2
df=pd.read_csv("/kaggle/input/salary-dataset-simple-linear-regression/Salary_dataset.csv")

#step 3
plt.scatter(df['YearsExperience'],df['Salary'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('scatter plot')
plt.show()

#step 4
X=df[['YearsExperience']]
Y=df['Salary']

#step 5
model=LinearRegression()
model.fit(X,Y)#calculates best fitting relation between x and y
slope=model.coef_[0]#represents coefficient of x in y=mx+c
intercept=model.intercept_#point where regression line crosses the x axis

#step 6
def linear_regression(x):
    return slope * x + intercept

#step 7
plt.scatter(df['YearsExperience'],df['Salary'],label='Data Points')
plt.plot(df['YearsExperience'],[linear_regression(xi) for xi in df['YearsExperience']],color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression')
plt.show()

#step 8
new_x=6
prediction=linear_regression(new_x)
print(f"Prediction for x={new_x}: {prediction}")
new_x=pd.DataFrame({"YearsExperience":[6]})
print(model.predict(new_x))
      
