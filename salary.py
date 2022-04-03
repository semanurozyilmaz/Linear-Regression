import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("Salary.csv")
df.head()

experience = df[['YearsExperience']]
salary = df[['Salary']]
x_train, x_test,y_train,y_test = train_test_split(experience,salary,test_size=0.33, random_state=0)

linear_reg = LinearRegression()
linear_reg.fit(x_train,y_train)

prediction = linear_reg.predict(x_test)

plt.figure()
plt.scatter(df.YearsExperience,df.Salary,color="blue")
plt.plot(x_test, linear_reg.predict(x_test),color="red")
plt.xlabel("Deneyim (Yıl)")
plt.ylabel("Maaş (TL)")
plt.title("Deneyim Maaş İlişkisi")
plt.grid(True)
plt.show()

print(f"Linear Regression R2: {r2_score(y_test,prediction)}")