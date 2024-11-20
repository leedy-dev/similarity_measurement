from sentence_transformers import SentenceTransformer, util

# BERT
# 문장 구조와 문맥을 모두 반영한 정교한 유사도 측정.
# 의미적 비교에 가장 적합.

# BERT 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# 사용자 입력
text1 = input("첫 번째 문장을 입력하세요: ")
text2 = input("두 번째 문장을 입력하세요: ")

# 문장 임베딩
embedding1 = model.encode(text1, convert_to_tensor=True)
embedding2 = model.encode(text2, convert_to_tensor=True)

# 코사인 유사도 계산
cosine_sim = util.pytorch_cos_sim(embedding1, embedding2)
print(f"BERT 기반 코사인 유사도: {cosine_sim.item():.2f}")
