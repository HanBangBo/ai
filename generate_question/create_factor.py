from langchain.prompts import PromptTemplate
import random
import json


def create_prompt(type_value):
    if type_value == "객관식":
        template = """
        너는 한국 뉴스 및 일반 상식을 기반으로 객관식 문제를 출제하는 AI야.
        아래의 예제처럼 문제를 만들어 줘.

        [예제 1]
        문제: "{example_questions[0]}"
        보기: {example_options[0]}
        정답: "{example_answers[0]}"

        [예제 2]
        문제: "{example_questions[1]}"
        보기: {example_options[1]}
        정답: "{example_answers[1]}"

        이제 아래 뉴스 본문을 참고하여 객관식 문제를 JSON 형식으로 생성해.
        정답의 근거가 되는 뉴스 본문 속 문장을 찾아 **explanation** 필드에 추가해야 해.

        뉴스 본문: {news_content}

        JSON 형식 예시:
        ```json
        {{
            "question": "문제 내용",
            "options": ["보기1", "보기2", "보기3", "보기4"],
            "answer": "정답 보기",
            "explanation": "news_content에서 해당 정답이 나온 문장 그대로 제공"
        }}
        ```
        """

        example_data = get_random_questions(type_value)  # 리스트 반환됨

        # 🚀 JSON 문자열 변환 없이 그대로 저장 (리스트 형태 유지)
        example_dict = {
            "example_questions": [q["example_questions"] for q in example_data],
            "example_options": [q["example_options"] for q in example_data],
            "example_answers": [q["example_answers"] for q in example_data]
        }

        prompt = PromptTemplate(
            input_variables=["news_content"],
            partial_variables=example_dict,  # 🚀 리스트 그대로 전달
            template=template
        )

    elif type_value == "주관식":
        template = """
        너는 한국 뉴스 및 일반 상식을 기반으로 주관식 문제를 출제하는 AI야.
        아래의 예제처럼 문제를 만들어 줘.

        [예제 1]
        문제: "{example_questions[0]}"
        정답: "{example_answers[0]}"

        [예제 2]
        문제: "{example_questions[1]}"
        정답: "{example_answers[1]}"

        이제 아래 뉴스 본문을 참고하여 주관식 문제를 JSON 형식으로 생성해.
        정답의 근거가 되는 뉴스 본문 속 문장을 찾아 **explanation** 필드에 추가해야 해.

        뉴스 본문: {news_content}

        JSON 형식 예시:
        ```json
        {{
            "question": "문제 내용",
            "answer": "정답 보기",
            "explanation": "news_content에서 해당 정답이 나온 문장 그대로 제공"
        }}
        ```
        """

        example_data = get_random_questions(type_value)  # 리스트 반환됨

        # 🚀 JSON 문자열 변환 없이 그대로 저장 (리스트 형태 유지)
        example_dict = {
            "example_questions": [q["example_questions"] for q in example_data],
            "example_answers": [q["example_answers"] for q in example_data]
        }

        prompt = PromptTemplate(
            input_variables=["news_content"],
            partial_variables=example_dict,  # 🚀 리스트 그대로 전달
            template=template
        )
    return prompt


def load_questions(filename: str, num_questions: int = 2):
    with open(filename, "r", encoding="utf-8") as f:
        questions = json.load(f)

    return random.sample(questions, min(num_questions, len(questions)))


def get_random_questions(type_value):
    if type_value == "객관식":
        return load_questions("multiple_choice.json", 2)
    elif type_value == "주관식":
        return load_questions("short_answer.json", 2)


def create_keywords(source_value, keyword):
    return

