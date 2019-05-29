from Read import data

#Missing Values
missing_value=data.isnull().sum()
print(missing_value)

#Visualization of missing value
import seaborn as sns
sns.heatmap(data.isnull(), yticklabels=False, cmap='viridis')
#Drop The Missing Value Rows
data=data.dropna()



#pip install missingno
#import missingno as msno
#msno.matrix(data)
