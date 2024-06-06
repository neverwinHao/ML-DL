# 对联邦党人文集进行分类
import re
import numpy as np


# 数据预处理
def preprocess_text(text):
    # 去除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 转换为小写
    text = text.lower()
    # 分词
    words = text.split()
    return words


# 计算高频词
def calculate(words, n):
    freq_dict = {}
    for word in words:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1
    key = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    return key


# 读取文件
with open('origin_article.txt', 'r') as file:
    origin_article = file.read()

# 下面将文章拆分为列表，索引X即代表第X篇文章
split_paper = origin_article.split('FEDERALIST No. ')
author_M = []
keywords_M = []
author_H = []
keywords_H = []
author_Guess = []
author_all = []
author_index = []

for i in range(1, 86):
    if "HAMILTON AND MADISON" in split_paper[i]:
        pass
    elif "HAMILTON OR MADISON" in split_paper[i]:
        author_Guess.append(split_paper[i])
        author_index.append(i)
    elif "HAMILTON" in split_paper[i]:
        author_H.append(split_paper[i])
    elif "MADISON" in split_paper[i]:
        author_M.append(split_paper[i])
author_all = author_H + author_M + author_Guess

# 提取H的高频词
words_H = preprocess_text(' '.join(author_H))
features_H = calculate(words_H, 250)
for word, freq in features_H:
    keywords_H.append(word)

# 提取M高频词
words_M = preprocess_text(' '.join(author_M))
features_M = calculate(words_M, 250)
for word, freq in features_M:
    keywords_M.append(word)

# 合并两个作者的高频词汇作为关键词
set_H = set(keywords_H)
set_M = set(keywords_M)
keywords = set_H.union(set_M)
# 先验概率计算
Pr_H = len(author_H) / len(author_H + author_M)
Pr_M = 1 - Pr_H

# 构建特征向量
keywords = list(set(keywords))

vector_guess = np.zeros(len(keywords))
vector_H = np.zeros(len(keywords))
vector_M = np.zeros(len(keywords))
vector_list = []

# 计算M的条件概率
for M_word in words_M:
    for i, word in enumerate(keywords):
        if M_word == word:
            vector_M[i] += 1
condition_M = vector_M / np.sum(vector_M)

# 计算H的条件概率
for H_word in words_H:
    for i, word in enumerate(keywords):
        if H_word == word:
            vector_H[i] += 1
condition_H = vector_H / np.sum(vector_H)

# 进行预测
for i in range(11):
    papers = preprocess_text(author_Guess[i])
    vector_guess = np.zeros(len(keywords))  # 创建一个与关键词数量相同的零向量
    for j in papers:
        for k, word in enumerate(keywords):
            if j == word:
                vector_guess[k] = 1  # 对每篇文章单独构建特征向量

    vector_list.append(vector_guess)  # 将特征向量添加到列表中

    # 下面开始计算概率
    guess_M = vector_list[i] * condition_M
    guess_H = vector_list[i] * condition_H
    print(guess_M)
    log_prob_M = 0
    log_prob_H = 0
    for pm in guess_M:
        if pm != 0:
            log_prob_M += np.log(pm)
        else:
            pass
    for ph in guess_H:
        if ph != 0:
            log_prob_H += np.log(ph)
        else:
            pass


    prob_M = log_prob_M + np.log(Pr_M)
    prob_H = log_prob_H + np.log(Pr_H)
    if prob_M > prob_H:
        print("第", author_index[i], "篇的作者为Madision")
    else:
        print("第", author_index[i], "篇的作者为Hamilton")
    print('--------------------------------------------')
