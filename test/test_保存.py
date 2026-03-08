
import pickle
import os
import numpy as np


# featurematrixnormalization = np.random.rand(1000, 1000)
# # 假设 featurematrixnormalization 已经被正确计算
#
# # 获取当前脚本的绝对路径
# script_dir = os.path.dirname(os.path.abspath("D:\研究项目相关代码\hnsw-python-master\minhash-lsh-hnsw20241015\test\test_保存.py"))
#
# # 定义文件存储路径，确保它位于项目文件夹内
# file_path = os.path.join(script_dir, 'featurematrixnormalization1.pkl')
#
# # 尝试保存 featurematrixnormalization 矩阵到项目文件夹中
# try:
#     with open(file_path, 'wb') as f:
#         pickle.dump(featurematrixnormalization, f)
#     print(f"Feature matrix normalization saved to {file_path}")
# except Exception as e:
#     print(f"An error occurred while saving the file: {e}")
#
# # 下次运行程序时，可以从项目文件夹中加载矩阵
# try:
#     with open(file_path, 'rb') as f:
#         featurematrixnormalization = pickle.load(f)
#     print(f"Feature matrix normalization loaded from {file_path}")
# except FileNotFoundError:
#     print(f"{file_path} not found. Generating from scratch...")
#     # 这里应该包含生成 featurematrixnormalization 的代码
# print(featurematrixnormalization)







script_dir = os.path.dirname(os.path.abspath("D:\研究项目相关代码\hnsw-python-master\minhash-lsh-hnsw20241015\test\test_保存.py"))

# # 定义文件存储路径，确保它位于项目文件夹内
file_path = os.path.join(script_dir, 'featurematrixnormalization_test.pkl')
# 检查文件是否存在
if os.path.exists(file_path):
    try:
        # 文件存在，直接加载
        with open(file_path, 'rb') as f:
            featurematrixnormalization = pickle.load(f)
        print(f"Feature matrix normalization loaded from {file_path}")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
else:
    # 文件不存在，生成 featurematrixnormalization 并保存
    featurematrixnormalization = np.random.rand(1000, 1000)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(featurematrixnormalization, f)
        print(f"Feature matrix normalization saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

# 输出 featurematrixnormalization
print(featurematrixnormalization)