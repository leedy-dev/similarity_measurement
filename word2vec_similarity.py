from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Word2Vec
# 단어 벡터의 평균을 사용하여 문장 간 의미적 유사성을 비교.
# 문맥 정보를 고려하지만, 문장의 구조는 반영하지 않음.

# 사전 학습된 Word2Vec 모델 로드 (Google Word2Vec 모델 예시)
# 모델 다운로드: https://code.google.com/archive/p/word2vec/
word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# 텍스트 전처리 함수
def preprocess(text):
    return [word for word in text.lower().split() if word in word2vec_model]

# 문장을 Word2Vec 벡터로 변환
def sentence_vector(sentence):
    words = preprocess(sentence)
    if not words:  # 문장이 비어 있을 경우
        return np.zeros(word2vec_model.vector_size)
    return np.mean([word2vec_model[word] for word in words], axis=0)

# 사용자 입력
text1 = input("첫 번째 문장을 입력하세요: ")
text2 = input("두 번째 문장을 입력하세요: ")

# 문장 벡터화
vector1 = sentence_vector(text1)
vector2 = sentence_vector(text2)

# 코사인 유사도 계산
similarity = cosine_similarity([vector1], [vector2])
print(f"Word2Vec 기반 코사인 유사도: {similarity[0][0]:.2f}")
