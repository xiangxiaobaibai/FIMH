import pandas as pd
import numpy as np
data= 'D:/生物序列检索_相关库/UNITE_public_25.07.2023/UNITE_public_25.07.2023.fasta'
path = 'D:/生物序列检索_相关库/'
def readfasta(lines):
    """读入并处理FASTA文件的函数"""
    seq = []
    index = []
    seqplast = ""
    numlines = 0

    for i in lines:
        if '>' in i:  # 判断是序列行还是说明行
            index.append(i.replace("\n", "").replace(">", ""))
            seq.append(seqplast.replace("\n", ""))
            seqplast = ""
            numlines += 1
        else:
            seqplast = seqplast + i.replace("\n", "")  # 把分行的序列拼接成一个字符串
            numlines += 1

        if numlines == len(lines):
            seq.append(seqplast.replace("\n", ""))

    seq = seq[1:]  # 移除第一个空序列
    return index, seq
f = open(data, 'r')
lines = f.readlines()
f.close()

(index, seq) = readfasta(lines)  # 接收序列名和序列
df = pd.DataFrame({'index':index,'seq':seq})
df['s'] = [ind.split('|')[2] for ind in df['index']]
# data cleaning
df =  df[~df['seq'].str.contains('[^ACGT]', na=False)]
df = df[~df['index'].str.contains('Incertae', na=False)]
df.shape
s_counts = df['s'].value_counts()
s_counts
# data with n_count > 15
sampled_data = df[df['s'].isin(s_counts[s_counts > 15].index)]
sampled_data.shape

#  train data:
# 计算每个类别的数量
s_counts = df['s'].value_counts()

# 获取数量大于等于15的类别
s_over_15 = s_counts[s_counts >= 15].index

# 使用groupby对每个类别进行采样，并设置replace=False和random_state=1
train_data = df[df['s'].isin(s_over_15)].groupby('s').apply(lambda x: x.sample(n=15, replace=False, random_state=1))

# 重置索引，因为groupby和apply操作会引入多级索引
train_data = train_data.reset_index(drop=True)

train_data.shape

train_data.to_csv(path + '15.csv')
train_data = pd.read_csv('./data/sampled_train_data.csv')
# train_data to fasta
with open(path + '15.fasta', 'w') as fasta_out:
    for species, seq in zip(sampled_data['index'], train_data['seq']):
        fasta_out.write(f'>{species}\n{seq}\n')

prefixes = ['k', 'p', 'c', 'o', 'f', 'g', 's']

for index, row in train_data.iterrows():
    taxon_info = row['index'].split('|')[1].split(';')

    for ind, (prefix, taxon) in enumerate(zip(prefixes, taxon_info)):
        if prefix != 's':
            train_data.at[index, prefix] = taxon.split('__')[-1]
        else:
            train_data.at[index, prefix] = row['index'].split('|')[-1]

train_data.drop(columns=['index','Unnamed: 0'], inplace=True)

cols = ['seq', 'k', 'p', 'c', 'o', 'f', 'g', 's']
train_data = train_data[cols]
train_data.head()

# train_data with kpcofgs
train_data.to_csv('./data/15.csv')






