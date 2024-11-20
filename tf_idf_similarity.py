from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF
# 단어의 빈도와 중요도를 기반으로 한 유사도.
# 짧은 텍스트에서 단어가 겹칠수록 높은 유사도를 가짐.

# 사용자 입력
text1 = input("첫 번째 문장을 입력하세요: ")
text2 = input("두 번째 문장을 입력하세요: ")

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text1, text2])

# 코사인 유사도 계산
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"TF-IDF 기반 코사인 유사도: {similarity[0][0]:.2f}")