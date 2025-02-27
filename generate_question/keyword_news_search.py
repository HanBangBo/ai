import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from datetime import datetime, timedelta


openai_api_key = os.getenv("OPENAI_API_KEY")
FAISS_PATH = "/mnt/efs_faiss_index/news_faiss"
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


# 검색 함수
def search_news(query, top_k=2, similarity_threshold=0.5, date_filter=None, press_filter=None, section_filter=None):
    # 1️⃣ 유사도 검색 먼저 수행 (최대 50개까지 가져와서 필터링)
    initial_k = max(top_k * 10, 50)
    docs_with_scores = vector_db.similarity_search_with_score(query, k=initial_k)

    # 2️⃣ 필터링 적용
    filtered_docs = []
    for doc, score in docs_with_scores:
        if score > similarity_threshold:
            continue  # 유사도 기준보다 낮으면 제외

        # 2-1️⃣ 날짜 필터링
        if date_filter:
            now = datetime.now()
            date_threshold = now - timedelta(days=30 * date_filter)
            try:
                doc_date = datetime.strptime(str(doc.metadata.get("date", "19700101")), "%Y%m%d")
                if doc_date < date_threshold:
                    continue  # 날짜 기준 미달이면 제외
            except ValueError:
                continue  # 날짜 형식이 올바르지 않으면 제외

        # 2-2️⃣ 언론사 필터링 (우선순위 적용)
        if press_filter and doc.metadata.get("press") != press_filter:
            continue  # 필터링된 언론사만 포함

        # 2-3️⃣ 카테고리 필터링
        if section_filter and doc.metadata.get("section") != section_filter:
            continue  # 카테고리 다르면 제외

        filtered_docs.append({"doc": doc, "score": score})  # ✅ Dictionary 형태로 저장하여 안전하게 정렬 가능

    # 3️⃣ 최소 개수 확보 (필터링 후 top_k보다 적으면 추가 검색)
    if len(filtered_docs) < top_k:
        additional_docs = [
            {"doc": doc, "score": score} for doc, score in docs_with_scores
            if doc not in [d["doc"] for d in filtered_docs]
        ]
        filtered_docs.extend(additional_docs[:top_k - len(filtered_docs)])

    # 4️⃣ 유사도 순 정렬
    filtered_docs.sort(key=lambda x: x["score"])  # ✅ 오류 없이 정렬 가능하도록 수정

    # ✅ **검색된 뉴스 본문만 합쳐서 반환 (중복 제거)**
    merged_text = " ".join(set(d["doc"].page_content for d in filtered_docs[:top_k]))

    return merged_text  # ✅ 본문만 반환
