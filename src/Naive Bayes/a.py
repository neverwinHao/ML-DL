import re
def preprocess_text(text):
    # 去除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 转换为小写
    text = text.lower()
    # 分词
    words = text.split()
    return words

def calculate(words, n):
    freq_dict = {}
    for word in words:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1
    key = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    return key

word = preprocess_text('I love love love china ss ni')
print(calculate(word, 5))
