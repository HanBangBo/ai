import requests
import time

url = "http://ec2-52-79-153-90.ap-northeast-2.compute.amazonaws.com:8000/generate_questions/"

params = {
    "type_value": "객관식",
    "source_value": "사회",
    "period": 1,
    "source_type": "카테고리",
    "keyword": {
        "탄핵": "60%",
        "돈": "50%"
    }
}

start_time = time.time()  # 요청 시작 시간
response = requests.post(url, json=params)  # GET이 아닌 POST 요청!
end_time = time.time()  # 요청 종료 시간

total_time = end_time - start_time  # 총 응답 시간 계산

print(f"응답 시간: {total_time:.3f} 초")
print(response.json())
