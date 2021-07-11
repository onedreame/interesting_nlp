




import pandas as pd
import numpy as np
fp = 'videoplay_predict.out'
data = pd.read_csv(fp, sep='\t', header=None)
columns = ['imp_id', 'predict', 'litectr_label', 'pctr']
data.columns = columns
data=data.drop_duplicates(subset=[columns[0]])
print(data.head())
ranges = np.linspace(0,1,4)
ps = data[columns[1]].quantile(ranges).tolist()
print(ranges)
print(ps)
percent = []
bias = []
vp = []
ss = []
print(max(data[columns[1]]))

print(data[data[columns[1]]<= 2.266980].shape[0]/data.shape[0])
positive_example = data[data[columns[-2]]==1]
negtive_example = data[data[columns[-2]]==0]
print(f'正例:{positive_example.shape[0]} 负例:{negtive_example.shape[0]}')
pctr = (sum(positive_example[columns[-1]]/1000000) + sum(negtive_example[columns[-1]]/1000000)*20)/(
    positive_example.shape[0] + 20* negtive_example.shape[0])
ctr = positive_example.shape[0]/ (positive_example.shape[0] + 20 * negtive_example.shape[0])
# pctr = sum(data[columns[-1]]/1000000)/data.shape[0]
# ctr = sum(data[columns[-2]])/data.shape[0]
print(f'总体bias')
print(pctr/ctr-1)
for idx in range(len(ranges)-1):
    sd = data[(data[columns[1]] < ps[idx+1]) & (data[columns[1]]>=ps[idx])]
    percent.append(ranges[idx+1])
    vp.append(ps[idx+1])
    ss.append(sd.shape[0])
    positive_example = sd[sd[columns[-2]] == 1]
    negtive_example = sd[sd[columns[-2]] == 0]
    pctr = (sum(positive_example[columns[-1]] / 1000000) + sum(negtive_example[columns[-1]] / 1000000) * 20) / (
            positive_example.shape[0] + 20 * negtive_example.shape[0])
    ctr = positive_example.shape[0] / (positive_example.shape[0] + 20 * negtive_example.shape[0])

    # pctr = sum(sd[columns[-1]] / 1000000) / sd.shape[0]
    # ctr = sum(sd[columns[-2]]) / sd.shape[0]
    bias.append(pctr / ctr - 1 if ctr > 0 else -1)

for i,j, k,v in zip(percent, vp, bias, ss):
    print(f'{i}\t{j}\t{k}\t{v}')

for idx in range(1, 20):
    sd = data[(data[columns[1]] < idx) & (data[columns[1]]>=idx-1)]
    if sd.shape[0]==0:
        print(f'error {idx} == 0')
        continue
    positive_example = sd[sd[columns[-2]] == 1]
    negtive_example = sd[sd[columns[-2]] == 0]
    pctr = (sum(positive_example[columns[-1]] / 1000000) + sum(negtive_example[columns[-1]] / 1000000) * 20) / (
            positive_example.shape[0] + 20 * negtive_example.shape[0])
    ctr = positive_example.shape[0] / (positive_example.shape[0] + 20 * negtive_example.shape[0])

    # pctr = sum(sd[columns[-1]] / 1000000) / sd.shape[0]
    # ctr = sum(sd[columns[-2]]) / sd.shape[0]
    bias.append(pctr / ctr - 1 if ctr > 0 else -1)
    print(f'{idx-1}-{idx}, bias: {bias[-1]}')

for idx in range(1, 20):
    sd = data[data[columns[1]] < idx]
    if sd.shape[0]==0:
        print(f'error {idx} == 0')
        continue
    positive_example = sd[sd[columns[-2]] == 1]
    negtive_example = sd[sd[columns[-2]] == 0]
    pctr = (sum(positive_example[columns[-1]] / 1000000) + sum(negtive_example[columns[-1]] / 1000000) * 20) / (
            positive_example.shape[0] + 20 * negtive_example.shape[0])
    ctr = positive_example.shape[0] / (positive_example.shape[0] + 20 * negtive_example.shape[0])

    # pctr = sum(sd[columns[-1]] / 1000000) / sd.shape[0]
    # ctr = sum(sd[columns[-2]]) / sd.shape[0]
    bias.append(pctr / ctr - 1 if ctr > 0 else -1)
    print(f'{0}-{idx}, bias: {bias[-1]}')