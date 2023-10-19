import pandas as pd
import numpy as np
import operator
import argparse

def KNN(IB2, df, k):

error_count = 0
for i in range (len(df)):

distance_dict = calculate_distances(IB2, df, i)
point_label = calculate_label(distance_dict, df, 0, 0, '', k)

if point_label != df['label'][i]:
error_count += 1
 
print(error_count)    
for i in range (len(IB2)):
print(IB2['label'][i],IB2['x'][i],IB2['y'][i],sep=',')


def calculate_distances(IB2, df, curr):
n = 0
distance_dict = {}

for i in range(len(IB2)):
p1 = df['x'][curr],df['y'][curr]
p2 = IB2['x'][i],IB2['y'][i]
dist = np.linalg.norm(np.array((p1))- np.array((p2)))
distance_dict [n] = [dist, IB2['label'][i]]
n+= 1

return distance_dict

def calculate_label(distance_dict, df, weight_A, weight_B, point_label, k):

distance_dict = sorted(distance_dict.items(), key=operator.itemgetter(1))  

#Nearest Neighbor
dist_1,label_1 = distance_dict[0][1][0],distance_dict[0][1][1]
#Farthest Neighbor
dist_k,label_k = distance_dict[k-1][1][0],distance_dict[k-1][1][1]

for i in range(k):

dist_i,label_i = distance_dict[i][1][0],distance_dict[i][1][1]

if dist_k == dist_1:
weight = 1
else:
weight = (dist_k - dist_i)/(dist_k - dist_1)


if label_i == 'A':
weight_A+= weight
else:
weight_B+= weight


if weight_A > weight_B:
point_label = 'A'

else:
point_label = 'B'
 
return point_label



def update_casebase(IB2, dataset, df):
point_label = ''

for i in range (len(dataset)):
for j in list(IB2.index):
p1 = dataset['x'][i],dataset['y'][i]
p2 = IB2['x'][j],IB2['y'][j]
curr_dist = np.linalg.norm(np.array((p1))-np.array((p2)))
if j == 0:
minimum = curr_dist
point_label = IB2['label'][j]
else:
if minimum > curr_dist:
minimum = curr_dist
point_label = IB2['label'][j]

if (point_label != df['label'][i]):
IB2=IB2.append(df.iloc[i])

return IB2, dataset, df


def main():
args = parser.parse_args()
data, k = args.data, int(args.k)
headers = ['label', 'x', 'y']
df= pd.read_csv(data,names = headers)
dataset = df[['x','y']]
IB2 = df.iloc[[0]]
IB2, dataset, df = update_casebase(IB2, dataset, df)

df = pd.concat([df,IB2]).drop_duplicates(keep=False)          
k=min(k,len(IB2))

IB2 = IB2.reset_index(drop=True)
df = df.reset_index(drop=True)


KNN(IB2, df, k)

if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument("--data",help="Location of Data File")
parser.add_argument("--k",help="Value of k for kNN classifier",type=int)
main()
