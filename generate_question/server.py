from fastapi import FastAPI
from typing import List
from generate_question.generate_question import generate_question_with_lang_chain
from generate_question.create_factor import create_keywords

app = FastAPI()


@app.get("/generate_questions/")
async def generate_questions(keyword: List[str], type_value: str, source_value: str, period: str):
    question = []
    k_value = create_keywords(source_value, keyword)

    for k in k_value:
        json_temp = generate_question_with_lang_chain(type_value, k, period, source_value)
        json_temp['keyword'] = k
        json_temp['type'] = type_value
        json_temp['source'] = source_value
        question.append(json_temp)

    return question
