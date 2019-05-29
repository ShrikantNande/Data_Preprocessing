from Datatype import num_cols, cat_cols
from Read import data
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

#Visulization
#Box_plot:- Numarical Values
for i in num_cols:
    print('Variable_name:',i)
    #Histogram Plot
    data[i].hist()
    plt.show()
    #Box Plot
    data[i].plot(kind='box', subplots=True, layout=(1,2), sharex=False, sharey=False)
    plt.show()
    #Density Plot
    data[i].plot.kde()
    plt.show()
 
   #Pie Chart    
    x_list = s_data[i]#Create list for column names
    label_list = s_data.index#lable list elements
    plt.axis("equal") #The pie chart is oval by default. To make it a circle use    
    pyplot.axis("equal")
    plt.pie(x_list,labels=label_list,autopct="%1.1f%%")#To show the percentage of each pieslice, pass an output format to the autopctparameter 
    plt.title("Pie chart")#Tital for the pie chart
    plt.show()#Show the plot

#Scatter Matrix Plot 
scatter_matrix(data, alpha=0.2, figsize=(16, 16), diagonal='kde')
plt.show()

#Bar_plot:-Categorical values
for i in cat_cols:
    print('Variable_name:',i)
    data[i].value_counts().head(10).plot.bar()
    plt.show()


