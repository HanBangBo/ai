import json
import re
import os
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from create_factor import create_prompt
from keyword_news_search import search_news
from langchain.schema.runnable import RunnableSequence


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"
llm = ChatOpenAI(model_name=MODEL, openai_api_key=OPENAI_API_KEY, max_tokens=512)


def generate_question_with_lang_chain(type_value, keyword, period, source, source_type):
    prompt_template = create_prompt(type_value)
    # LangChain을 이용하여 뉴스 내용을 기반으로 문제를 생성
    if source_type == "언론사":
        news_content = search_news(keyword, top_k=3, similarity_threshold=0.5, date_filter=period, press_filter=source)
    elif source_type == "카테고리":
        news_content = search_news(keyword, top_k=3, similarity_threshold=0.5, date_filter=period, section_filter=source)

    # RunnableSequence 적용
    question_chain = prompt_template | llm  # RunnableSequence 사용

    # 문제 생성
    response = question_chain.invoke({"news_content": news_content})
    if hasattr(response, "content"):
        response_text = response.content  # AIMessage에서 텍스트 추출
    else:
        response_text = str(response)  # 안전장치 (혹시라도 오류 방지)

    # JSON 변환
    json_data = extract_json_from_text(response_text)
    return json_data


def extract_json_from_text(text):
    try:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)  # JSON 패턴 추출
        if json_match:
            json_str = json_match.group(0)  # 정규식으로 찾은 JSON 문자열
            return json.loads(json_str)  # JSON 변환
        else:
            raise ValueError("JSON 데이터가 없습니다.")

    except json.JSONDecodeError as e:
        print(f"JSON 변환 오류: {e}")
        print(text)
        return None


# ✅ 실행 예제
if __name__ == "__main__":
    q1 = generate_question_with_lang_chain(type_value="주관식", keyword="엔비디아", period=6, source="한국경제", source_type="언론사")
    print(q1)
    q1 = generate_question_with_lang_chain(type_value="주관식", keyword="엔비디아", period=6, source="정치", source_type="카테고리")
    print(q1)
