import os
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# OpenAI API Key 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
save_path = "//mnt/efs_faiss_index//keyword_faiss"

# FAISS 인덱스 로드
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_db = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)


def create_keywords(keyword, source_type, source_value):
    """
    ✅ 사용자 맞춤형 키워드 리스트를 생성하는 함수

    Parameters:
        - params (dict): 백엔드에서 넘어온 데이터
          - user (str): 사용자 ID
          - source_type (str): '언론사' 또는 '카테고리'
          - source_value (list): 선택된 언론사 or 카테고리 리스트
          - keyword (dict): {키워드: 오답률} 형태의 딕셔너리

    Returns:
        - list: 최종 10개 키워드 리스트
    """

    additional_keywords = []

    # ✅ FAISS 벡터DB에서 필터링하여 키워드 가져오기
    if source_type == "언론사":
        results = vector_db.similarity_search("", k=100, filter={"press": source_value})
        filtered_results = [
            doc.page_content for doc in results if doc.metadata.get("press") == source_value
        ]
        print(f"📌 언론사 '{source_value}'에서 가져온 키워드 개수: {len(filtered_results)}")  # ✅ 디버깅
        additional_keywords.extend(filtered_results)

    elif source_type == "카테고리":
        results = vector_db.similarity_search("", k=100, filter={"section": source_value})
        filtered_results = [
            doc.page_content for doc in results if doc.metadata.get("section") == source_value
        ]
        print(f"📌 카테고리 '{source_value}'에서 가져온 키워드 개수: {len(filtered_results)}")  # ✅ 디버깅
        additional_keywords.extend(filtered_results)

    # ✅ 키워드 빈도수 계산 후 가장 많이 나온 키워드 추출
    keyword_counts = Counter(additional_keywords)
    sorted_keywords = [kw for kw, _ in keyword_counts.most_common()]


    final_keyword_list = []
    for key in keyword:
        final_keyword_list.append(key)

    # ✅ 10개 이상 채우기
    for kw in sorted_keywords:
        if len(final_keyword_list) < 10 and kw not in keyword:
            final_keyword_list.append(kw)

    if len(final_keyword_list) < 10:
        extra_results = vector_db.similarity_search("", k=200)  # 더 많은 데이터 검색
        extra_keywords = [doc.page_content for doc in extra_results]
        extra_keyword_counts = Counter(extra_keywords)
        sorted_extra_keywords = [kw for kw, _ in extra_keyword_counts.most_common()]

        for kw in sorted_extra_keywords:
            if len(final_keyword_list) < 10 and kw not in final_keyword_list:
                final_keyword_list.append(kw)

    return final_keyword_list[:10]
