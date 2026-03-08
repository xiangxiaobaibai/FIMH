
import pprint
import sys
import time
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from operator import itemgetter
from random import random
import numpy as np
import pickle
import os
import math
import random
from Bio import SeqIO
import faiss
from collections import defaultdict



#读取1000条序列
def read_fasta_sequences(fasta_file, num_sequences=1000):
    sequences = []
    count = 0
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq).upper())
        count += 1
        if count >= num_sequences:
            break
    return sequences

# 指定地址并且读取序列
fasta_file = "D:\生物序列检索_相关库\sh_general_release_dynamic_25.07.2023.fasta"
sequences = read_fasta_sequences(fasta_file, 10)

#创建shingles
def build_shingles(sentence: str, k: int):
    shingles = []
    for i in range(len(sentence) - k):
        shingles.append(sentence[i:i+k])
    return shingles

#创建vocab(其由shingles构成)
def build_vocab(shingle_sets: list):
    # 将shingle集合列表转换为单个集合
    full_set = {item for set_ in shingle_sets for item in set_}
    vocab = {}
    for i, shingle in enumerate(list(full_set)):
        vocab[shingle] = i
    return vocab

#创建one-hot编码
def one_hot(shingles: set, vocab: dict):
    vec = np.zeros(len(vocab))
    for shingle in shingles:
        idx = vocab[shingle]
        vec[idx] = 1
    return vec

#创建"""线性同余散列函数"""
def linear_congruential_hash(a, c, m, x):
    return ((a * x + c) % m)

#生成minhash特征
def generate_minhash_signature(shingles_1hot, k):
    """生成MinHash签名"""
    n_features = len(shingles_1hot)

    global sss  # 声明sss是一个全局变量
    print(sss)  # 打印当前值
    sss += 1  # 增加sss的值

    # signature = np.full(k, np.inf)  # 初始化签名数组，用无穷大填充
    signature = [np.inf] * k
    # 生成k个线性同余散列函数的参数
    a_values = np.random.randint(1, high=n_features/1000, size=k)
    c_values = np.random.randint(0, high=n_features/1000, size=k)
    m_values = np.random.randint(n_features, high=2 * n_features, size=k)

    # a_values = np.random.randint(1, high=n_features, size=k)
    # c_values = np.random.randint(0, high=n_features, size=k)
    # m_values = np.random.randint(n_features, high=2 * n_features, size=k)
    # 对每个哈希函数进行操作
    for i in range(k):
        a, c, m = a_values[i], c_values[i], m_values[i]
        min_hash_value = np.inf
        for j, shingle in enumerate(shingles_1hot):
            if shingle == 1:
                hash_value = linear_congruential_hash(a, c, m, j)
                if hash_value < min_hash_value:
                    min_hash_value = hash_value
        signature[i] = min_hash_value
    return signature

from itertools import combinations






#此函数用于对各个文件中的内容进行k-shingle，然后对词条进行哈希（此处就用字典存储了），其中dir是文件夹的名称字符串类型，k是int型
def getShingleList(dir, k):
    fileList = getFilesName(dir)
    shingleList = list()
    for fileName in fileList:
        fileContent = getFileContent(fileName)
        shingle = set()
        for index in range(0, len(fileContent) - k + 1):
            shingle.add(fileContent[index:index + k])
        shingleList.append(shingle)
    return shingleList




#此处是新版的函数，将哈希签名的矩阵换的行列换了一下，便于接下来使用
def getMinHashSignature(shingleList, signatureNum):
    # tatalSet用于存放所有集合的并集,shingleList: 一个列表，其中每个元素是一个集合，代表一个文档的 shingles
    totalSet = shingleList[0]
    for i in range(1, len(shingleList)):
        totalSet = set(totalSet) | set(shingleList[i])

    temp = int(math.sqrt(signatureNum))
    # randomArray用于模拟随机哈希函数
    randomArray = []
    # signatureList用于存放总的哈希签名
    signatureList = []
    g=1
    maxNum = sys.maxsize
    for i in range(signatureNum):
        randomArray.append(random.randint(1, temp * 2))
        randomArray.append(random.randint(1, temp * 2))
    # buketNum用于记录所有元素的个数，作为随机哈希函数的桶号
    buketNum = len(totalSet)
    """
    此处将不同文档的自己的哈希签名存成一个list，然后再进行汇总到一个总的list
    """
    for shingleSet in shingleList:
        """
        signature用于存放哈希函数产生的签名
        """
        signature = []
        for i in range(signatureNum):
            minHash = maxNum
            for index, item in enumerate(totalSet):
                if item in shingleSet:
                    num = (randomArray[i * 2] * index + randomArray[i * 2 + 1]) % buketNum
                    minHash = min(minHash, num)
            signature.append(minHash)
        signatureList.append(signature)
        print("提取序列特征数量"+str(g))
        g=g+1
    return signatureList







#此函数通过比较两个文档的最小哈希签名进行计算相似度，传入的参入是两个文档的最小哈希签名的集合，存放在list中，最后结果返回相似度
def calSimilarity(signatureSet1, signatureSet2):
    count = 0
    for index in range(len(signatureSet1)):
        if (signatureSet1[index] == signatureSet2[index]):
            count += 1
    return count / (len(signatureSet1) * 1.0)


#此函数用于将计算所有文档的相似度，并将结果存放在一个list中，结果用元组存放
def calAllSimilarity(signatureList, filesName):
    signatureNum = len(signatureList)
    fileNum = len(filesName)
    result = []
    for index1, signatureSet1 in enumerate(signatureList):
        for index2, signatureSet2 in enumerate(signatureList):
            if (index1 < index2):
                result.append((calSimilarity(signatureSet1, signatureSet2), filesName[index1], filesName[index2]))
    return result






















# 定义LSH类，这个LSH类的目的是将签名分割成子向量，并将这些子向量存储在哈希桶中，以便于快速查找具有相似哈希值的签名，从而找到可能的相似项。
class LSH:
    buckets = []
    counter = 0
    # 构造函数，接受一个参数b，表示将签名分成多少个桶（band）
    def __init__(self, b):
        self.b = b
        for i in range(b):
            self.buckets.append({})
    # 接受一个签名，并将其划分为多个子向量。
    def make_subvecs(self, signature):
        l = len(signature)
        assert l % self.b == 0
        r = int(l / self.b)
        # break signature into subvectors
        subvecs = []
        for i in range(0, l, r):
            subvecs.append(signature[i:i + r])
        return np.stack(subvecs)
    # 将签名添加到哈希桶中
    def add_hash(self, signature):
        subvecs = self.make_subvecs(signature).astype(str)
        for i, subvec in enumerate(subvecs):
            subvec = ','.join(subvec)
            if subvec not in self.buckets[i].keys():
                self.buckets[i][subvec] = []
            self.buckets[i][subvec].append(self.counter)
        self.counter += 1
    def check_candidates(self):
        candidates = {}
        for bucket_band in self.buckets:
            keys = bucket_band.keys()
            for bucket in keys:
                hits = bucket_band[bucket]
                if len(hits) > 1:
                    # 生成所有可能的候选者对
                    pairs = combinations(hits, 2)
                    for pair in pairs:
                        # 使用候选者对作为键，更新它们出现的次数
                        candidates[pair] = candidates.get(pair, 0) + 1
        return candidates


#判断模块，看看是不是已经生成了特征矩阵feature_matrix_normalization_ys，生成了就不用再生成了
script_dir = os.path.dirname(os.path.abspath("D:\研究项目相关代码\hnsw-python-master\minhash-lsh-hnsw20241015\test\test_保存.py"))
# 定义文件存储路径，确保它位于项目文件夹内
file_path = os.path.join(script_dir, 'feature_matrix_normalization_ys.pkl')
# 检查文件是否存在
if os.path.exists(file_path):
    try:
        # 文件存在，直接加载
        with open(file_path, 'rb') as f:
            feature_matrix_normalization_ys = pickle.load(f)
        print(f"Feature matrix normalization loaded from {file_path}")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
else:
    # 文件不存在，生成 featurematrixnormalization 并保存
    # shingle 尺寸
    k = 9
    # 建立shingles
    shingles = []
    for sequence in sequences:
        shingles.append(build_shingles(sequence, k))
    print(shingles)

    start_time = time.time()



    signatureList = getMinHashSignature(shingles, 200)

    stop_time = time.time()

    pprint.pprint("k=,本次提取搜索时间为: %f" % (start_time - stop_time))
    print(signatureList)
    print(1)
    print(len(signatureList[0]))

#def calSimilarity(signatureSet1, signatureSet2):

    # 初始化相似性矩阵，大小为 signatureList 长度的平方
    similarityMatrix = [[1 for _ in range(len(signatureList))] for _ in range(len(signatureList))]

    print(similarityMatrix)

    # 遍历 signatureList，计算每个元素与其他元素之间的相似度
    for i in range(len(signatureList)):
        for j in range(len(signatureList)):
            if i != j:
                # 计算相似度并存储在相似性矩阵中
                similarity = calSimilarity(signatureList[i], signatureList[j])
                similarityMatrix[i][j] = similarity
    # 现在相似性矩阵中包含了 signatureList 中所有元素之间的相似度
    print(similarityMatrix)
    # 将生成的特征signature也存入文件
    script_dir1 = os.path.dirname(
        os.path.abspath("D:\研究项目相关代码\hnsw-python-master\minhash-lsh-hnsw20241015\test\test_保存.py"))
    # 定义文件存储路径，确保它位于项目文件夹内
    file_path1 = os.path.join(script_dir1, 'signature_ys.pkl')
    try:
        with open(file_path1, 'wb') as f:
            pickle.dump(signatureList, f)
        print(f"Signature saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
    print(signatureList)


    #将特征矩阵存储到文件中
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(similarityMatrix, f)
        print(f"Feature matrix normalization saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")