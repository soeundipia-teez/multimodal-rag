import os
import time
import pickle
import base64
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

processed = PROJECT_ROOT / "data" / "processed"
figures_dir = PROJECT_ROOT / "data" / "figures"


# ============================================================
# 2. 텍스트 및 표 요약 함수
# ============================================================

def generate_text_summaries(
    texts,
    tables,
    summarize_texts=False
):
    """
    텍스트 및 표 데이터를 요약합니다.

    texts: 텍스트 청크 리스트
    tables: 표 리스트
    summarize_texts: 텍스트 요약 여부
    """

    prompt_text_kor = """
    당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록
    돕는 도우미입니다.

    이 요약본들은 임베딩되어 원본 텍스트나 표 요소를
    검색하는 데 사용될 것입니다.

    주어진 표나 텍스트의 내용을 검색에 최적화된
    간결한 한국어 요약으로 작성해주세요.

    경제전망의 연도, 수치, 단위를 유지해주세요.
    자료에 없는 정보는 만들지 마세요.

    요약할 표 또는 텍스트: {element}
    """

    prompt = ChatPromptTemplate.from_template(
        prompt_text_kor
    )

    model = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini"
    )

    # 실습에서 학습한 LCEL 체인
    summarize_chain = (
        {"element": lambda x: x}
        | prompt
        | model
        | StrOutputParser()
    )

    text_summaries = []
    table_summaries = []

    # 텍스트 요약
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(
            texts,
            config={"max_concurrency": 2}
        )

    elif texts:
        text_summaries = texts

    # 표 요약
    if tables:
        table_summaries = summarize_chain.batch(
            tables,
            config={"max_concurrency": 2}
        )

    return text_summaries, table_summaries


# ============================================================
# 3. 이미지 Base64 인코딩
# ============================================================

def encode_image(image_path):
    """이미지를 Base64 문자열로 변환합니다."""

    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


# ============================================================
# 4. 이미지 요약 함수
# ============================================================

def image_summarize(
    img_base64,
    prompt,
    mime_type="image/png"
):
    """
    GPT-4o-mini를 활용하여 이미지 요약을 생성합니다.
    """

    chat = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=1024
    )

    msg = chat.invoke([
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{mime_type};base64,"
                            f"{img_base64}"
                        )
                    }
                }
            ]
        )
    ])

    return msg.content


# ============================================================
# 5. 전체 이미지 요약 함수
# ============================================================

def generate_img_summaries(path, selected_files):
    """
    이번 과제 PDF에서 지정한 이미지만 요약합니다.

    selected_files:
        extracted.pkl에 저장된 figure_names

    figures 폴더에 다른 JPG가 남아 있어도 제외합니다.
    """

    img_base64_list = []
    image_summaries = []
    image_file_names = []

    if not selected_files:
        raise ValueError(
            "figure_names가 없습니다. "
            "extract.py의 추출 결과를 확인하세요."
        )

    if len(selected_files) != len(set(selected_files)):
        raise ValueError(
            "figure_names에 중복 이미지가 있습니다."
        )

    prompt_kor = """
    You are an assistant tasked with summarizing images
    for retrieval.

    These summaries will be embedded and used to retrieve
    the original image.

    Describe the chart title, axes, years, units,
    key numbers and trends accurately.

    Do not invent information not visible in the image.

    The summary must be written in Korean (Hangul).

    Image filename: {filename}
    """

    print(
        "이번 PDF 분석 대상 이미지:",
        len(selected_files),
        "개"
    )

    # 폴더 전체가 아니라 이번 PDF 이미지 목록만 순회
    for img_file in sorted(selected_files):

        if not img_file.lower().endswith(".png"):
            continue

        img_path = os.path.join(
            path,
            img_file
        )

        if not os.path.isfile(img_path):
            raise FileNotFoundError(
                f"이미지 파일이 없습니다: {img_path}"
            )

        base64_image = encode_image(img_path)

        img_base64_list.append(base64_image)
        image_file_names.append(img_file)

        summary = image_summarize(
            base64_image,
            prompt_kor.format(filename=img_file),
            mime_type="image/png"
        )

        # 파일명을 요약문에 포함해 검색 시 출처 식별
        image_summaries.append(
            f"원본 파일: {img_file}\n{summary}"
        )

        print(
            f"이미지 요약 완료: {img_file}"
        )

    return (
        img_base64_list,
        image_summaries,
        image_file_names
    )


# ============================================================
# 6. 메인 실행
# ============================================================

def main():

    # --------------------------------------------------------
    # 6-1. API 키 로딩
    # --------------------------------------------------------

    load_dotenv(PROJECT_ROOT / ".env")

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY가 없습니다. "
            ".env 파일을 확인하세요."
        )

    # --------------------------------------------------------
    # 6-2. 기존 추출 결과 불러오기
    # --------------------------------------------------------

    pkl_path = processed / "extracted.pkl"

    if not pkl_path.is_file():
        raise FileNotFoundError(
            "extracted.pkl이 없습니다. "
            "먼저 extract.py를 실행하세요."
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    texts = data["texts"]
    tables = data["tables"]
    texts_2k_token = data["texts_2k_token"]

    # 이번 과제 PDF에서 생성된 PNG 목록
    figure_names = data.get("figure_names", [])

    print("texts 개수:", len(texts))
    print("tables 개수:", len(tables))
    print(
        "texts_2k_token 개수:",
        len(texts_2k_token)
    )
    print(
        "추출 대상 이미지 수:",
        len(figure_names)
    )

    # --------------------------------------------------------
    # 6-3. 텍스트 및 표 요약
    # --------------------------------------------------------

    start = time.time()

    text_summaries, table_summaries = (
        generate_text_summaries(
            texts_2k_token,
            tables,
            summarize_texts=True
        )
    )

    print(
        f"\n텍스트·표 요약 완료 => "
        f"{time.time() - start:.1f}초"
    )

    print("텍스트 요약 수:", len(text_summaries))
    print("표 요약 수:", len(table_summaries))

    if text_summaries:
        print("\n[텍스트 요약 예시]")
        print(text_summaries[0][:300])

    if table_summaries:
        print("\n[표 요약 예시]")
        print(table_summaries[0][:300])

    # --------------------------------------------------------
    # 6-4. 이미지 요약
    # --------------------------------------------------------

    start = time.time()

    (
        img_base64_list,
        image_summaries,
        image_file_names
    ) = generate_img_summaries(
        str(figures_dir),
        figure_names
    )

    print(
        f"\n이미지 요약 완료 => "
        f"{time.time() - start:.1f}초"
    )

    print(
        "이미지 요약 수:",
        len(image_summaries)
    )

    if not image_summaries:
        raise RuntimeError(
            "이미지 요약 결과가 없습니다. "
            "figures 폴더를 확인하세요."
        )

    print("\n[이미지 요약 예시]")
    print(str(image_summaries[0])[:500])

    # --------------------------------------------------------
    # 6-5. 요약 결과 저장
    # --------------------------------------------------------

    output_file = processed / "summaries.pkl"

    with open(output_file, "wb") as f:
        pickle.dump(
            {
                "text_summaries": text_summaries,
                "table_summaries": table_summaries,
                "image_summaries": image_summaries,
                "img_base64_list": img_base64_list,
                "image_file_names": image_file_names
            },
            f
        )

    print("\n저장 완료:", output_file)
    print(
        "다음 실행: python src/retriever.py"
    )


if __name__ == "__main__":
    main()
