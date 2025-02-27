import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from datetime import datetime, timedelta


openai_api_key = os.getenv("OPENAI_API_KEY")
FAISS_PATH = "/mnt/efs_faiss_index/news_faiss"
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


# 검색 함수
def search_news(query, top_k=3, similarity_threshold=0.4, date_filter=None, press_filter=None, section_filter=None):
    # FAISS 유사도 검색
    docs_with_scores = vector_db.similarity_search_with_score(query, k=top_k)

    # 유사도 필터링
    filtered_docs = [(doc, score) for doc, score in docs_with_scores if score <= similarity_threshold]

    if not filtered_docs:
        print("검색된 뉴스가 유사도 기준을 충족하지 않습니다.")
        return None, None

    # 날짜 필터링 적용
    if date_filter:
        # 현재 날짜
        now = datetime.now()

        # 날짜 필터링 기준 설정
        if date_filter == "1":
            date_threshold = now - timedelta(days=30)  # 최근 1개월
        elif date_filter == "3":
            date_threshold = now - timedelta(days=90)  # 최근 3개월
        elif date_filter == "6":
            date_threshold = now - timedelta(days=180)  # 최근 6개월
        else:
            raise ValueError("지원되지 않는 날짜 필터입니다. ('1', '3', '6' 중 선택)")

        # 날짜 비교하여 필터링
        def is_recent(doc):
            doc_date = datetime.strptime(doc.metadata["date"], "%Y-%m-%d")
            return doc_date >= date_threshold

        filtered_docs = [(doc, score) for doc, score in filtered_docs if is_recent(doc)]

        if not filtered_docs:
            print(f"'{date_filter}' 이내의 뉴스가 없습니다.")
            return None, None

    # 언론사(press) 필터링 적용
    if press_filter:
        filtered_docs = [(doc, score) for doc, score in filtered_docs if doc.metadata["press"] == press_filter]

        if not filtered_docs:
            print(f"'{press_filter}'의 기사가 없습니다.")
            return None, None

    # 카테고리(section) 필터링 적용
    if section_filter:
        filtered_docs = [(doc, score) for doc, score in filtered_docs if doc.metadata["section"] == section_filter]

        if not filtered_docs:
            print(f"'{press_filter}'의 기사가 없습니다.")
            return None, None

    # 유사도 순 정렬
    filtered_docs.sort(key=lambda x: x[1])  # score 기준으로 정렬 (낮을수록 유사)

    # 검색된 뉴스 출력
    print("\n검색된 뉴스 유사도 점수:")
    for idx, (doc, score) in enumerate(filtered_docs):
        print(f"  {idx + 1}. {doc.metadata['press']} - {doc.metadata['date']} (유사도: {score:.4f})")

    # 뉴스 본문 통합 (중복 제거)
    unique_texts = list(set(doc.page_content for doc, _ in filtered_docs))
    merged_text = " ".join(unique_texts).replace("\n", " ")

    print("\n뉴스 본문 통합 결과:")
    print(merged_text)

    return filtered_docs, merged_text  # 검색된 뉴스 데이터와 본문 반환

