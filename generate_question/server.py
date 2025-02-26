from fastapi import FastAPI
from typing import List
from generate_question.generate_question import generate_question_with_lang_chain, return_keyword

app = FastAPI()


@app.get("/generate_questions/")
async def generate_questions(keyword: List[str], type_value: str, source_value: str, date: str):
    question = []
    keywords = return_keyword(source_value, keyword)

    for k in keywords:
        json_temp = generate_question_with_lang_chain(type_value, k, date, source_value)
        json_temp['keyword'] = k
        json_temp['type'] = type_value
        json_temp['source'] = source_value
        question.append(json_temp)

    return question
