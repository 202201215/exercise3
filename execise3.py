import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# 从古腾堡数据集读取白鲸档案
whale_text = gutenberg.raw('melville-moby_dick.txt')

# 标记化
tokens = word_tokenize(whale_text)

# 停止词过滤
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# 词性(POS)标记
pos_tags = nltk.pos_tag(filtered_tokens)

# POS频率
pos_counts = Counter(tag for word, tag in pos_tags)
top_pos = pos_counts.most_common(5)

# 词序化
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens[:20]]

# 绘制频率分布
pos_tags, counts = zip(*pos_counts.items())
x_pos = np.arange(len(pos_tags))

plt.figure(figsize=(8, 5))  # 调整图表尺寸

plt.bar(x_pos, counts, align='center', width=0.5)
plt.xticks(x_pos, pos_tags, rotation=90)  # 旋转横坐标标签
plt.xlabel("POS Tags")
plt.ylabel("Frequency")
plt.title("POS Tag Frequency Distribution")
plt.tight_layout()
plt.show()