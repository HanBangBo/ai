import json
import re
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from fastapi import FastAPI
from typing import List
from generate_question.create_prompt import create_prompt


app = FastAPI()
OPENAI_API_KEY = "api_key"
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)


@app.get("/generate_questions/")
async def generate_questions(keyword: List[str], type_value: str, source_value: str, date: str):
    question = []  # 여러 개의 JSON을 담을 리스트

    keywords = return_keyword(source_value, keyword)
    for k in keywords:
        json_temp = generate_question_with_lang_chain(type_value, k, date, source_value)
        json_temp['keyword'] = k
        json_temp['type'] = type_value
        json_temp['source'] = source_value
        question.append(json_temp)
    return question


def generate_question_with_lang_chain(type_value, keyword, date, source):
    # LangChain을 이용하여 뉴스 내용을 기반으로 문제를 생성
    prompt_template = create_prompt(type_value)
    news_content = create_news_content(keyword, date, source)

    # ✅ LangChain 최신 방식 적용
    question_chain = LLMChain(llm=llm, prompt=prompt_template)

    response = question_chain.invoke({"news_content": news_content})
    json_text = extract_json(response["text"])
    return json.loads(json_text)


def return_keyword(source, keyword):
    return [""]


def extract_json(text):
    # 응답에서 JSON 부분만 추출하는 함수
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        raise ValueError("JSON 데이터를 찾을 수 없습니다.")


def create_news_content(keyword, date):
    """기존 뉴스 데이터 그대로 유지"""
    content = """
    윤석열 대통령은 25일 헌법재판소에서 열린 탄핵심판 최종변론에서 직접 최후진술을 했다. 이는 헌정 사상 처음 있는 일이다.
    윤 대통령은 12.3 비상계엄 선포가 거대 야당의 국정 마비 시도에 대응하기 위한 '대국민 호소'였으며, 자신을 위한 결정이 아니었다고 주장했다.
    비상계엄은 과거 군사 정권의 계엄과 다르며, 국가 위기를 국민에게 알리고자 했다고 강조했다.
    국회에 투입된 병력은 최소한으로 유지했으며, 내란을 일으킬 의도가 없었다고 주장했다.
    탄핵소추 이후 국민과 청년들이 나라를 지키기 위해 나섰다며, 자신이 직무에 복귀하면 개헌과 정치개혁에 집중하겠다고 밝혔다.

    윤 대통령은 탄핵심판 최종변론에서 12.3 비상계엄이 야당의 국정 운영 방해에 대응하기 위한 것이었다고 주장하며, 이를 '대국민 호소'라고 표현했다.
    계엄 선포가 국가를 위한 결정이었으며, 군사 쿠데타와 같은 의도가 없었음을 강조했다.
    국회의 예산·입법권 남용을 비판하며, 정부 운영을 방해하는 야당의 태도를 강하게 질타했다.
    탄핵심판에서 자신이 복귀할 경우, 남은 임기에 연연하지 않고 개헌을 추진하겠다고 선언했다. 이는 임기 단축을 전제로 한 정치개혁 의지를 내비친 것으로 해석된다.
    국민들에게 계엄으로 인한 혼란과 불편을 끼친 점에 대해 사과했으며, 헌법재판소에 자신의 결단을 고려해 줄 것을 요청했다.
    법조계에서는 헌재의 최종 선고가 3월 초중순쯤 이뤄질 것으로 전망하고 있다.
    """
    return content
