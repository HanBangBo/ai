from langchain.prompts import PromptTemplate
from typing import List
import random
import json


def create_prompt(type_value):
    example_data = get_random_questions(type_value)  # ✅ 문제 데이터 로드

    example_dict = {
        "example_questions": [q["example_questions"] for q in example_data],
        "example_options": [q["example_options"] for q in example_data] if type_value == "객관식" else None,
        "example_answers": [q["example_answers"] for q in example_data]
    }

    # ✅ JSON에서 중괄호 `{}`가 템플릿 변수가 되지 않도록 문자열로 처리
    json_template = '''
        {{
            "question": "문제 내용",
            "options": ["보기1", "보기2", "보기3", "보기4"],
            "answer": "정답 보기",
            "explanation": "news_content에서 해당 정답이 나온 문장 그대로 제공"
        }}
    '''

    if type_value == "객관식":
        template = f"""
        다음 뉴스 데이터를 바탕으로 전체적인 트렌드를 평가할 수 있는 객관식 문제를 생성하세요.
        세부 숫자보다 정의나 설명을 반영한 질문을 1개만 만드세요.
        아래의 예제처럼 문제를 만들어 줘.
        보기는 되도록 단답형으로 만들어 줘.

        [예제 1]
        문제: "{example_dict["example_questions"][0]}"
        보기: {example_dict["example_options"][0]}
        정답: "{example_dict["example_answers"][0]}"

        [예제 2]
        문제: "{example_dict["example_questions"][1]}"
        보기: {example_dict["example_options"][1]}
        정답: "{example_dict["example_answers"][1]}"

        이제 아래 뉴스 본문을 참고하여 객관식 문제를 JSON 형식으로 생성해.
        정답의 근거가 되는 뉴스 본문 속 문장을 찾아 **explanation** 필드에 추가해야 해.

        뉴스 본문: {{news_content}}

        JSON 형식 예시:
        {json_template}
        """.strip()

    elif type_value == "주관식":
        json_template = '''
        {{
            "question": "문제 내용",
            "answer": "정답 보기",
            "explanation": "news_content에서 해당 정답이 나온 문장 그대로 제공"
        }}
        '''
        template = f"""
        다음 뉴스 데이터를 바탕으로 전체적인 트렌드를 평가할 수 있는 주관식 문제를 생성하세요.
        세부 숫자보다 정의나 설명을 반영한 질문을 1개만 만들어 줘.
        아래의 예제처럼 문제를 만들어 줘.
        정답은 너무 길지 않은은 단답형으로 만들어 줘줘.

        [예제 1]
        문제: "{example_dict["example_questions"][0]}"
        정답: "{example_dict["example_answers"][0]}"

        [예제 2]
        문제: "{example_dict["example_questions"][1]}"
        정답: "{example_dict["example_answers"][1]}"

        이제 아래 뉴스 본문을 참고하여 주관식 문제를 JSON 형식으로 생성해.
        정답의 근거가 되는 뉴스 본문 속 문장을 찾아 **explanation** 필드에 추가해야 해.

        뉴스 본문: {{news_content}}

        JSON 형식 예시:
        {json_template}
        """.strip()

    return PromptTemplate(
        input_variables=["news_content"],  # ✅ input으로 "news_content"만 남김
        template=template
    )


def load_questions(filename: str, num_questions: int = 2):
    with open(filename, "r", encoding="utf-8") as f:
        questions = json.load(f)

    return random.sample(questions, min(num_questions, len(questions)))


def get_random_questions(type_value):
    if type_value == "객관식":
        return load_questions("multiple_choice.json", 2)
    elif type_value == "주관식":
        return load_questions("short_answer.json", 2)
