import pprint
import sys
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from operator import itemgetter
from random import random
import numpy as np
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
sequences = read_fasta_sequences(fasta_file, 1000)
#创建shingles
def build_shingles(sentence: str, k: int):
    shingles = []
    for i in range(len(sentence) - k):
        shingles.append(sentence[i:i+k])
    return set(shingles)

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

import numpy as np
# shingle 尺寸
k = 5
# 建立shingles
shingles = []
for sequence in sequences:
    shingles.append(build_shingles(sequence, k))

# 建立 vocab
vocab = build_vocab(shingles)

# 用one-hot对shingles进行编码
shingles_1hot = []
for shingle_set in shingles:
    shingles_1hot.append(one_hot(shingle_set, vocab))

# 堆叠成一个单一的NumPy数组
shingles_1hot = np.stack(shingles_1hot)
shingles_1hot.shape

def linear_congruential_hash(a, c, m, x):
    """线性同余散列函数"""
    return ((a * x + c) % m)

def generate_minhash_signature(shingles_1hot, k):
    """生成MinHash签名"""
    n_features = len(shingles_1hot)
    #     signature = np.full(k, np.inf)  # 初始化签名数组，用无穷大填充

    signature = [np.inf] * k

    # 生成k个线性同余散列函数的参数
    a_values = np.random.randint(1, high=n_features, size=k)
    c_values = np.random.randint(0, high=n_features, size=k)
    m_values = np.random.randint(n_features, high=2 * n_features, size=k)

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



# 定义LSH类，这个LSH类的目的是将签名分割成子向量，
# 并将这些子向量存储在哈希桶中，以便于快速查找具有相似哈希值的签名，从而找到可能的相似项。
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
b = 500#签名是500个数字，这边分成了500个桶，所以每个签名的字向量长度为1个数字
# 请求用户输入一系列序列
user_input = input("请输入你要搜索的序列: ")
# s="CCGTGGGGATTCGTCCCCATTGAGATAGCACCCTTTGTTCATGAGTACCCTCGTTTCCTCGGCGGGCTCGCCCGCCAGCAGGACAACTTCAAACCCTTTGCAGTAGCAGTAACTTCAGTTAATAACAAATATTAAAACTTTCAACAACGGATCTCTTGGTTCTGGCATCGATGAAGAACGCAGCGAAATGCGATAAGTAGTGTGAATTGCAGAATTCAGTGAATCATCGAATCTTTGAACGCACATTGCGCCCTTCGGTATTCCGTTGGGCATGCCTGTTCGAGCGTCATTTAAACCTTCAAGCTATGCTTGGTGTTGGGTGTCTGTCCCGCCTCAGCGCGTGGACTCGCCTCAAATCCATTGGCAGCCGGTATGTTGGCTTCGTGCGCAGCACATTGCAAGCGGAACCATCAGACCCCCTCCC"
# 将序列其转换成数字签名
# num_shingles_1hot = len(shingles_1hot)
# k = 500  # 假设我们想要500个哈希函数
# signature = []
# for i in range(num_shingles_1hot):
#     signature.append(generate_minhash_signature(shingles_1hot[i], k))
# print(signature)
# shingle 尺寸
# user_input = list(user_input)
k_user = 5
# 建立shingles
shingles_user = []
shingles_user.append(build_shingles(user_input, k_user))
vocab_user = build_vocab(shingles_user)
# 用one-hot对shingles进行编码
shingles_1hot_user = []
for shingle_set in shingles_user:
    shingles_1hot_user.append(one_hot(shingle_set, vocab_user))
# 堆叠成一个单一的NumPy数组
shingles_1hot_user = np.stack(shingles_1hot_user)
shingles_1hot_user.shape

# # 对每个哈希函数进行操作
# for i in range(k):
#     a, c, m = a_values[i], c_values[i], m_values[i]
#     min_hash_value = np.inf
#     for j, shingle in enumerate(shingles_1hot_user):
#         if shingle == 1:
#             hash_value = linear_congruential_hash(a, c, m, j)
#             if hash_value < min_hash_value:
#                 min_hash_value = hash_value
#     signature[i] = min_hash_value
# return signature
# 使用
num_shingles_1hot_user = len(shingles_1hot_user)
k2 = 500  # 假设我们想要500个哈希函数
signature_user = []
for i in range(num_shingles_1hot_user):
    signature_user.append(generate_minhash_signature(shingles_1hot_user[i], k2))
print(signature_user)
print(signature_user[0])
print(len(signature_user[0]))

list= [2, 6, 4, 4, 2, 1, 3, 3, 6, 1, 7, 1, 0, 6, 14, 3, 4, 1, 2, 1, 3, 1, 1, 3, 20, 3, 0, 1, 0, 0, 2, 2, 14, 6, 6, 5, 0, 8, 3, 6, 4, 9, 7, 2, 2, 1, 0, 4, 9, 1, 2, 0, 6, 2, 2, 7, 2, 2, 4, 0, 6, 2, 0, 0, 5, 9, 4, 9, 1, 18, 1, 7, 1, 0, 10, 3, 2, 0, 5, 3, 5, 0, 23, 0, 1, 3, 1, 5, 1, 3, 1, 5, 0, 10, 0, 5, 4, 2, 4, 2, 3, 1, 1, 4, 3, 1, 4, 0, 8, 16, 2, 0, 0, 5, 2, 1, 0, 4, 4, 0, 4, 0, 3, 7, 2, 0, 0, 2, 2, 7, 5, 14, 3, 1, 6, 1, 1, 3, 3, 4, 1, 0, 1, 2, 12, 9, 1, 0, 10, 13, 9, 10, 11, 2, 0, 3, 4, 0, 1, 3, 0, 1, 0, 36, 2, 9, 1, 0, 2, 0, 7, 3, 6, 5, 172, 0, 0, 12, 7, 3, 10, 2, 2, 12, 0, 0, 3, 5, 6, 1, 39, 1, 103, 1, 1, 1, 1, 7, 1, 8, 0, 21, 1, 1, 7, 6, 2, 1, 7, 3, 11, 6, 4, 1, 2, 5, 0, 5, 14, 6, 0, 0, 11, 3, 1, 1, 7, 0, 11, 3, 3, 5, 4, 3, 4, 0, 1, 1, 3, 0, 3, 2, 0, 4, 6, 0, 2, 0, 0, 5, 3, 8, 5, 6, 2, 1, 0, 2, 3, 2, 0, 1, 5, 3, 8, 2, 3, 5, 3, 2, 14, 4, 0, 8, 4, 0, 1, 12, 1, 7, 11, 2, 2, 1, 1, 1, 2, 9, 3, 5, 0, 6, 0, 0, 2, 14, 9, 2, 0, 0, 14, 1, 13, 5, 1, 2, 5, 14, 18, 3, 4, 2, 0, 0, 2, 0, 4, 6, 5, 5, 4, 0, 0, 22, 2, 6, 0, 8, 3, 0, 2, 7, 4, 6, 0, 3, 2, 1, 2, 1, 1, 2, 5, 8, 4, 1, 0, 4, 0, 1, 2, 3, 2, 4, 0, 0, 3, 2, 2, 1, 7, 3, 2, 0, 1, 0, 1, 5, 1, 6, 0, 3, 0, 2, 0, 0, 21, 2, 1, 5, 3, 1, 6, 5, 5, 11, 13, 16, 1, 0, 11, 2, 7, 2, 2, 6, 0, 8, 5, 3, 10, 0, 15, 11, 8, 8, 3, 14, 0, 3, 1, 3, 3, 2, 18, 5, 1, 0, 1, 1, 5, 3, 2, 7, 14, 6, 4, 3, 20, 5, 7, 4, 2, 26, 7, 5, 4, 4, 4, 7, 6, 15, 10, 1, 2, 5, 2, 12, 3, 3, 17, 4, 5, 0, 5, 4, 6, 2, 3, 5, 2, 0, 0, 2, 1, 0, 5, 4, 0, 1, 2, 3, 3, 3, 4, 4, 5, 0, 4, 0, 1, 3, 3, 2, 0, 3, 3, 2, 0, 7, 7, 0, 19, 9, 1, 3, 0, 0, 1, 4]
print(len(list))





