import os
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import faiss

import streamlit as st

# 경로 설정
data_path = './data'
module_path = './modules'


# Gemini 설정
import google.generativeai as genai

# import shutil
# os.makedirs("/root/.streamlit", exist_ok=True)
# shutil.copy("secrets.toml", "/root/.streamlit/secrets.toml")
# 어렵다 어려워...z..
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key="AIzaSyDIYgGSW1JnXWtqmPjSA7YR9-A4xz7-YYg")

# Gemini 모델 선택
model = genai.GenerativeModel("gemini-1.5-flash")


# CSV 파일 로드
## 자체 전처리를 거친 데이터 파일 활용
csv_file_path = "JEJU_MCT_DATA_modified.csv"
df = pd.read_csv(os.path.join(data_path, csv_file_path))

# 최신연월 데이터만 가져옴
df = df[df['기준연월'] == df['기준연월'].max()].reset_index(drop=True)


# Streamlit App UI

st.set_page_config(page_title="🍊참신한 제주 맛집!")

# Replicate Credentials
with st.sidebar:
    st.title("🍊참신한! 제주 맛집")

    st.write("")
     
    st.subheader("언드레 가신디가?")

    # selectbox 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stSelectbox label {  /* This targets the label element for selectbox */
            display: none;  /* Hides the label element */
        }
        .stSelectbox div[role='combobox'] {
            margin-top: -20px; /* Adjusts the margin if needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    time = st.sidebar.selectbox("", ["아침", "점심", "오후", "저녁", "밤"], key="time")

    st.write("")

    st.subheader("어드레가 맘에 드신디가?")

    # radio 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stSelectbox label {  /* This targets the label element for selectbox */
            display: none;  /* Hides the label element */
        }
        .stSelectbox div[role='combobox'] {
            margin-top: -20px; /* Adjusts the margin if needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    place = st.sidebar.selectbox("", ["제주", "서귀포", "성산"], key="place")

    st.write("")


    st.write("")

     # radio 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stRadio > label {
            display: none;
        }
        .stRadio > div {
            margin-top: -20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.title("혼저 옵서예!👋")
st.subheader("군맛난 제주 밥집🧑‍🍳 추천해드릴게예")

st.write("")

st.write("#흑돼지 #갈치조림 #옥돔구이 #고사리해장국 #전복뚝배기 #한치물회 #빙떡 #오메기떡..🤤")

st.write("")

image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRTHBMuNn2EZw3PzOHnLjDg_psyp-egZXcclWbiASta57PBiKwzpW5itBNms9VFU8UwEMQ&usqp=CAU"
image_html = f"""
<div style="display: flex; justify-content: center;">
    <img src="{image_path}" alt="centered image" width="50%">
</div>
"""
st.markdown(image_html, unsafe_allow_html=True)

st.write("")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "어드런 식당 찾으시쿠과?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어드런 식당 찾으시쿠과?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# RAG

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face의 사전 학습된 임베딩 모델과 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

print(f'Device is {device}.')


# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index.index')):
    """
    FAISS 인덱스를 파일에서 로드합니다.

    Parameters:
    index_path (str): 인덱스 파일 경로.

    Returns:
    faiss.Index: 로드된 FAISS 인덱스 객체.
    """
    if os.path.exists(index_path):
        # 인덱스 파일에서 로드
        index = faiss.read_index(index_path)
        print(f"FAISS 인덱스가 {index_path}에서 로드되었습니다.")
        return index
    else:
        raise FileNotFoundError(f"{index_path} 파일이 존재하지 않습니다.")

# 텍스트 임베딩
def embed_text(text):
    # 토크나이저의 출력도 GPU로 이동
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        # 모델의 출력을 GPU에서 연산하고, 필요한 부분을 가져옴
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()  # 결과를 CPU로 이동하고 numpy 배열로 변환

# 임베딩 로드
embeddings = np.load(os.path.join(module_path, 'embeddings_array_file.npy'))

### Generator 영역 ###

def generate_response_with_faiss(question, df, embeddings, model, embed_text, time, local_choice, index_path=os.path.join(module_path, 'faiss_index.index'), max_count=10, k=3, print_prompt=True):
    # 데이터 프레임 초기화
    filtered_df = df
    
    # FAISS 인덱스를 로드하고 쿼리 임베딩 생성
    index = load_faiss_index(index_path)
    query_embedding = embed_text(question).reshape(1, -1)
    
    # 유사한 텍스트 검색
    distances, indices = index.search(query_embedding, k * 3)
    filtered_df = filtered_df.iloc[indices[0, :]].copy().reset_index(drop=True)

    # 시간대 필터링 조건 추가
    if time == '아침':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(5, 12)))].reset_index(drop=True)
    elif time == '점심':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(12, 14)))].reset_index(drop=True)
    elif time == '오후':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(14, 18)))].reset_index(drop=True)
    elif time == '저녁':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(18, 23)))].reset_index(drop=True)
    elif time == '밤':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in [23, 24, 1, 2, 3, 4]))].reset_index(drop=True)

    # 필터링 후 데이터가 없으면 알림 메시지 반환
    if filtered_df.empty:
        return f"현재 선택하신 시간대({time})에는 영업하는 가게가 없습니다."

    # 상위 k개의 결과만 사용
    filtered_df = filtered_df.reset_index(drop=True).head(k)

    # 현지인 맛집인지 관광객 맛집인지 필터링 추가
    if local_choice == '제주도민 맛집':
        local_choice = '제주도민(현지인) 맛집'
    elif local_choice == '관광객 맛집':
        local_choice = '현지인 비중이 낮은 관광객 맛집'

    # 사용자 맞춤형 추천 정보 생성
    reference_info = ""
    for idx, row in filtered_df.iterrows():
        # 기본 가게 정보 구성
        store_name = row['가맹점명']
        location = row['가맹점주소']
        category = row['가맹점업종']
        opening_date = row['가맹점개설일자']
        usage_amount_range = row['이용금액구간']
        local_ratio = row['현지인이용건수비중']

        # 추천 정보 포맷
        reference_info += f"**{store_name}**\n위치: {location}\n업종: {category}\n개업일: {opening_date}\n"
        reference_info += f"가격대: {usage_amount_range}\n"

        # 현지인 비율을 통해 현지인 맛집인지 관광객 맛집인지 설명 추가
        if local_ratio > 0.5:
            reference_info += "이곳은 현지인들이 자주 찾는 맛집입니다. 현지인 비율이 높아 진정한 제주 맛을 즐길 수 있는 곳으로 유명합니다.\n"
        else:
            reference_info += "이곳은 관광객들에게 인기가 많습니다. 제주에서 편안히 관광을 즐기며 음식을 즐길 수 있는 곳입니다.\n"

        # 메뉴 및 기타 정보
        reference_info += f"주요 메뉴 및 특징: {row['text']}\n\n"

    # 프롬프트 생성
    prompt = f"질문: {question} 특히 {local_choice}을 선호해\n참고할 정보:\n{reference_info}\n응답:"

    # 프롬프트가 잘 생성되었는지 확인
    if print_prompt:
        print('-----------------------------'*3)
        print(prompt)
        print('-----------------------------'*3)

    # 모델을 통해 응답 생성
    response = model.generate_content(prompt)
    return response

# User-provided prompt
if prompt := st.chat_input(): # (disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # response = generate_llama2_response(prompt)
            response = generate_response_with_faiss(prompt, df, embeddings, model, embed_text, time, local_choice)
            placeholder = st.empty()
            full_response = ''

            # 만약 response가 GenerateContentResponse 객체라면, 문자열로 변환하여 사용합니다.
            if isinstance(response, str):
                full_response = response
            else:
                full_response = response.text  # response 객체에서 텍스트 부분 추출

            # for item in response:
            #     full_response += item
            #     placeholder.markdown(full_response)

            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)