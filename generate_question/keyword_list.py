import pandas as pd
from collections import Counter
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
openai_api_key = os.getenv("OPENAI_API_KEY")

save_path = '/Users/sondain/Desktop/hakaton/keyword_faiss/index.faiss'
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vector_db = FAISS.load_local(save_path, embeddings)

def generate_user_keywords(user_id, user_keywords_data, keywords_df, press=None, category=None):
    """
    ✅ 사용자 맞춤형 키워드 리스트를 생성하는 함수

    Parameters:
        - user_id (str): 사용자 ID
        - user_keywords_data (DataFrame): 유저별 틀린 문제 키워드 데이터
        - news_data_df (DataFrame): 뉴스 데이터 (키워드 포함)
        - press (str, optional): 사용자가 선택한 언론사
        - category (str, optional): 사용자가 선택한 카테고리

    Returns:
        - list: 최종 10개 키워드 리스트
    """


    # ✅ 1️⃣ 사용자 틀린 문제 키워드 가져오기
    user_data = user_keywords_data[user_keywords_data["user"] == user_id]
    incorrect_keywords = user_data.sort_values("incorrect_count", ascending=False)["keyword"].tolist()

    # ✅ 2️⃣ 사용자가 선택한 언론사 & 카테고리 키워드 가져오기
    additional_keywords = []
    
    if press:
        press_keywords = keywords_df[keywords_df["press"] == press]["keywords"].explode().tolist()
        additional_keywords.extend(press_keywords)

    if category:
        category_keywords = keywords_df[keywords_df["category"] == category]["keywords"].explode().tolist()
        additional_keywords.extend(category_keywords)

    # ✅ 3️⃣ 키워드 빈도수 계산 및 추가
    keyword_counts = Counter(additional_keywords)
    sorted_keywords = [kw for kw, _ in keyword_counts.most_common()]

    # ✅ 4️⃣ 최종 키워드 리스트 구성 (10개 이상 보장)
    final_keyword_list = incorrect_keywords.copy()
    
    for kw in sorted_keywords:
        if len(final_keyword_list) < 10 and kw not in final_keyword_list:
            final_keyword_list.append(kw)

    return final_keyword_list[:10]

