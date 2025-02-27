import os
import json
import traceback
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# error log file
ERROR_LOG_FILE = "error_log.txt"
BATCH_SIZE = 100
FAISS_PATH = "faiss_path"

# OpenAI API Key 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


def load_or_create_faiss_index(save_path):
    if os.path.exists(save_path) and os.path.isdir(save_path):
        print(f"기존 FAISS 인덱스를 불러옵니다: {save_path}")
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"새로운 FAISS 인덱스를 생성합니다: {save_path}")
        return None  # 기존 인덱스가 없으면 새로 생성할 수 있도록 None 반환


# 뉴스 데이터 FAISS 저장/업데이트 함수
def create_or_update_news_faiss(news_articles, save_path=FAISS_PATH + "news_faiss"):
    # 기존 FAISS 인덱스를 로드
    news_faiss = load_or_create_faiss_index(save_path)

    existing_entries = set()

    if news_faiss:
        # 기존 뉴스 문서 가져오기
        docs = news_faiss.similarity_search("", k=1000)  # FAISS에서 최대 1000개 검색
        existing_entries = {
            (doc.page_content, doc.metadata["press"], doc.metadata["date"], doc.metadata["keyword"],
             doc.metadata["section"])
            for doc in docs
        }
        print(f"기존 뉴스 개수: {len(existing_entries)}")

    # 중복을 제거한 새로운 뉴스 리스트 생성
    new_news_documents = [
        Document(
            page_content=f"{article['content']}",  # 제목 + 본문 결합
            metadata={"press": article["press"], "date": article["date"], "keyword": article["keyword"],
                      "section": article["section"]}
        )
        for article in news_articles
        # 중복 방지: 기존 벡터 + 메타데이터 조합이 있는지 확인
        if (f"{article['content']}", article["press"], article["date"], article["keyword"],
            article["section"]) not in existing_entries
    ]

    if new_news_documents:
        if news_faiss:
            news_faiss.add_documents(new_news_documents)  # 기존 인덱스에 추가
        else:
            news_faiss = FAISS.from_documents(new_news_documents, embeddings)  # 새 인덱스 생성

        news_faiss.save_local(save_path)
        print(f"{len(new_news_documents)}개의 새로운 뉴스가 추가되었습니다.")
    else:
        print("모든 뉴스가 이미 존재합니다. 추가할 뉴스가 없습니다.")


# 키워드 FAISS 저장/업데이트 함수
def create_or_update_keyword_faiss(news_articles, save_path=FAISS_PATH + "keyword_faiss"):
    # 기존 FAISS 인덱스 로드 (없으면 None)
    keyword_faiss = load_or_create_faiss_index(save_path)

    existing_entries = set()

    if keyword_faiss:
        # 기존 키워드 + 메타데이터 목록 가져오기
        docs = keyword_faiss.similarity_search("", k=1000)  # 전체 검색
        existing_entries = {(doc.page_content, doc.metadata["press"], doc.metadata["date"], doc.metadata["section"]) for
                            doc in docs}
        print(f"기존 키워드 개수: {len(existing_entries)}")

    # 중복 제거하여 새로운 키워드 리스트 생성
    new_keywords = [
        Document(
            page_content=article["keyword"],
            metadata={"press": article["press"], "date": article["date"], "section": article["section"]}
        )
        for article in news_articles
        # 키워드 + 메타데이터 중복 방지
        if (article["keyword"], article["press"], article["date"], article["section"]) not in existing_entries
    ]

    if new_keywords:
        if keyword_faiss:
            keyword_faiss.add_documents(new_keywords)  # 기존 인덱스에 추가
        else:
            keyword_faiss = FAISS.from_documents(new_keywords, embeddings)  # 새 인덱스 생성

        keyword_faiss.save_local(save_path)
        print(f"{len(new_keywords)}개의 새로운 키워드가 추가되었습니다!")
    else:
        print("모든 키워드(메타데이터 포함)가 이미 존재합니다. 추가할 내용이 없습니다.")


# 오류 로그 기록 함수
def log_error(batch_index, article, error):
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[Batch {batch_index}] 오류 발생!\n")
        f.write(f"언론사: {article.get('press', 'N/A')}\n")
        f.write(f"날짜: {article.get('date', 'N/A')}\n")
        f.write(f"오류 메시지: {error}\n")
        f.write(f"{traceback.format_exc()}\n")
    print(f"오류 발생! Batch {batch_index} - 기사: {article.get('title', 'N/A')} (로그 저장됨)")


# JSON 파일을 불러와서 Batch 저장하는 함수
def process_news_json(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            news_articles = json.load(f)

        total_articles = len(news_articles)
        print(f"총 {total_articles}개의 뉴스 기사를 처리합니다.")

        for i in range(0, total_articles, BATCH_SIZE):
            batch = news_articles[i:i + BATCH_SIZE]
            batch_index = i // BATCH_SIZE + 1
            print(f"\n[Batch {batch_index}] {i + 1}번 ~ {i + len(batch)}번 기사 처리 중...")

            try:
                create_or_update_news_faiss(batch)
                create_or_update_keyword_faiss(batch)
                print(f"[Batch {batch_index}] 저장 완료!")
            except Exception as e:
                for article in batch:
                    log_error(batch_index, article, str(e))

    except Exception as e:
        log_error("전체", {}, str(e))
        print(f"전체 JSON 파일 처리 중 오류 발생! {str(e)}")


# 실행 예제
if __name__ == "__main__":
    json_file_path = "news_data.json"  # JSON 파일 경로
    process_news_json(json_file_path)
