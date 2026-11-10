import os
import re
import json
import uuid
import time
import pickle
import textwrap
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from PIL import Image, ImageDraw, ImageFont


# LangChain 버전별 import 경로 대응
try:
    from langchain_classic.retrievers.multi_vector import (
        MultiVectorRetriever
    )
except ImportError:
    from langchain.retrievers.multi_vector import (
        MultiVectorRetriever
    )

try:
    from langchain_core.stores import InMemoryStore
except ImportError:
    from langchain.storage import InMemoryStore


# ============================================================
# 1. 경로 및 질문 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

processed = PROJECT_ROOT / "data" / "processed"
results_dir = PROJECT_ROOT / "results"

id_key = "doc_id"

QUESTIONS = [
    "Table 1에서 2025년 실질 GDP 성장률, 실업률, PCE 물가상승률 전망 중앙값은 각각 얼마인가요?",

    "Figure 2의 점도표에서 2024년과 2025년 연말 적정 연방기금금리 전망 점들의 분포를 비교해 주세요.",

    "Figure 1에서 2024~2027년 GDP 성장률과 실업률 중앙값 전망은 어떻게 달라지나요?",

    "Table 2에 나타난 2027년 GDP 성장률과 실업률 전망의 역사적 예측 오차범위는 무엇인가요?",

    "Figure 4.A에서 2024~2027년 GDP 성장률 전망의 불확실성 범위를 그래프 기준으로 설명해 주세요."
]

# Before/After 비교 질문
query = QUESTIONS[1]


# ============================================================
# 2. 요약과 원본을 같은 ID로 저장
# ============================================================

def add_documents(
    retriever,
    doc_summaries,
    doc_contents,
    kind="text",
    filenames=None
):
    """
    요약문은 vectorstore에 저장하고,
    원본은 docstore에 같은 doc_id로 저장합니다.
    """

    if len(doc_summaries) != len(doc_contents):
        raise ValueError(
            f"{kind}: 요약 수와 원본 수가 다릅니다."
        )

    if not doc_contents:
        return

    filenames = filenames or [""] * len(doc_contents)

    if len(filenames) != len(doc_contents):
        raise ValueError(
            f"{kind}: 파일명 수가 원본 수와 다릅니다."
        )

    doc_ids = [
        str(uuid.uuid4())
        for _ in doc_contents
    ]

    # 검색용 요약 Document
    summary_docs = [
        Document(
            page_content=str(summary),
            metadata={
                id_key: doc_ids[i],
                "kind": kind,
                "filename": filenames[i]
            }
        )
        for i, summary in enumerate(doc_summaries)
    ]

    # 답변 생성에 사용할 원본 Document
    original_docs = [
        Document(
            page_content=content,
            metadata={
                "kind": kind,
                "filename": filenames[i]
            }
        )
        for i, content in enumerate(doc_contents)
    ]

    # 요약 → 벡터 DB
    retriever.vectorstore.add_documents(
        summary_docs
    )

    # 원본 → 문서 저장소
    retriever.docstore.mset(
        list(zip(doc_ids, original_docs))
    )


# ============================================================
# 3. 검색 결과를 이미지와 텍스트로 분리
# ============================================================

def split_image_text_types(docs):
    """
    검색된 원본을 이미지와 텍스트/표로 구분합니다.
    """

    b64_images = []
    texts_out = []
    sources = []

    for doc in docs:

        content = (
            doc.page_content
            if isinstance(doc, Document)
            else str(doc)
        )

        metadata = (
            doc.metadata
            if isinstance(doc, Document)
            else {}
        )

        kind = metadata.get("kind", "")

        # Base64 여부 또는 metadata로 이미지 판단
        is_image = (
            kind == "image"
            or bool(
                re.fullmatch(
                    r"[A-Za-z0-9+/]{100,}={0,2}",
                    content
                )
            )
        )

        if is_image:

            # API 비용과 입력 크기 제한
            if len(b64_images) >= 4:
                continue

            b64_images.append(content)
            kind = "image"

        else:
            texts_out.append(content)
            kind = kind or "text"

        sources.append({
            "type": kind,
            "file": metadata.get("filename", "")
        })

    return {
        "images": b64_images,
        "texts": texts_out,
        "sources": sources
    }


# ============================================================
# 4. 멀티모달 프롬프트 생성
# ============================================================

def img_prompt_func(data_dict):
    """
    원본 이미지와 텍스트를
    멀티모달 LLM이 읽을 수 있게 조립합니다.
    """

    messages = []

    # 4-1. 이미지 원본 전달
    for image in data_dict["context"]["images"]:

        messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image}"
            }
        })

    # 4-2. 텍스트/표 원본
    formatted_texts = "\n".join(
        data_dict["context"]["texts"]
    )

    # 4-3. 검색된 자료 목록
    sources = "\n".join(
        (
            f"- {item['type']}: "
            f"{item.get('file') or '페이지 정보 없음'}"
        )
        for item in data_dict["context"].get(
            "sources", []
        )
    )

    messages.append({
        "type": "text",
        "text": (
            "당신은 미국 연방준비제도의 2024년 12월 "
            "경제전망 보고서를 분석하는 AI입니다.\n\n"

            "아래 지침을 지키세요.\n"

            "1. 제공된 자료만 근거로 한국어로 답하세요.\n"

            "2. 표와 그래프의 연도, 숫자, 단위를 "
            "정확하게 구분하세요.\n"

            "3. 그래프의 Y축 전체 범위와 실제 점들이 "
            "분포한 범위를 혼동하지 마세요.\n"

            "4. Figure 2의 금리 전망 범위를 설명할 때 "
            "Table 1의 Range 값이 제공되어 있다면 "
            "함께 비교하세요.\n"

            "5. 보이지 않거나 확인할 수 없는 수치를 "
            "추측하지 마세요.\n"

            "6. 출처 페이지는 검색 근거에 "
            "명시된 경우에만 표시하세요.\n"

            "7. 근거가 부족하면 확인할 수 없다고 "
            "답하세요.\n\n"

            f"질문: {data_dict['question']}\n\n"

            f"검색된 원본 목록:\n{sources}\n\n"

            f"참고 텍스트:\n{formatted_texts}"
        )
    })

    return [
        HumanMessage(content=messages)
    ]


# ============================================================
# 5. Before/After 이미지 생성
# ============================================================

def make_comparison_png(before, after):
    """
    실제 두 RAG 답변을 이미지로 저장합니다.
    """

    font_path = Path(
        "C:/Windows/Fonts/malgun.ttf"
    )

    if font_path.exists():
        font_title = ImageFont.truetype(
            str(font_path), 27
        )
        font_body = ImageFont.truetype(
            str(font_path), 18
        )
    else:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    image = Image.new(
        "RGB",
        (1400, 1050),
        "white"
    )

    draw = ImageDraw.Draw(image)

    draw.text(
        (30, 24),
        "BEFORE: TEXT-ONLY RAG",
        fill="black",
        font=font_title
    )

    draw.text(
        (730, 24),
        "AFTER: MULTIMODAL RAG",
        fill="black",
        font=font_title
    )

    draw.line(
        (700, 18, 700, 1020),
        fill="#bbbbbb",
        width=2
    )

    def wrap(text, max_chars=34):
        result = []

        for paragraph in text.splitlines():
            result.extend(
                textwrap.wrap(
                    paragraph,
                    max_chars,
                    break_long_words=True
                ) or [""]
            )

        return result[:43]

    for x, content in [
        (30, before),
        (730, after)
    ]:

        for i, line in enumerate(
            wrap(content)
        ):
            draw.text(
                (x, 92 + i * 21),
                line,
                fill="black",
                font=font_body
            )

    image.save(
        results_dir / "before_after.png"
    )


# ============================================================
# 6. README 질의응답 표 자동 갱신
# ============================================================

def update_readme(qa_records):
    """
    README에서 지정된 Q&A 영역을
    실제 모델 출력으로 갱신합니다.
    """

    readme = PROJECT_ROOT / "README.md"

    if not readme.exists():
        return

    original = readme.read_text(
        encoding="utf-8"
    )

    start_tag = "<!-- QA_RESULTS_START -->"
    end_tag = "<!-- QA_RESULTS_END -->"

    if (
        start_tag not in original
        or end_tag not in original
    ):
        print(
            "README에 Q&A 갱신 태그가 없습니다."
        )
        return

    def sanitize(value):
        return (
            str(value)
            .replace("|", "\\|")
            .replace("\r", " ")
            .replace("\n", " ")
        )

    rows = [
        "| 번호 | 질문 | 답변 요약 | 검색된 원본 종류 |",
        "|---|---|---|---|"
    ]

    for item in qa_records:

        source_text = ", ".join(
            (
                f"{source['type']} "
                f"({source['file']})"
                if source["file"]
                else source["type"]
            )
            for source in item["sources"]
        )

        rows.append(
            f"| {item['number']} "
            f"| {sanitize(item['question'])} "
            f"| {sanitize(item['answer'])[:240]} "
            f"| {sanitize(source_text)} |"
        )

    before = original.split(start_tag)[0]
    after = original.split(end_tag)[1]

    readme.write_text(
        before
        + start_tag
        + "\n"
        + "\n".join(rows)
        + "\n"
        + end_tag
        + after,
        encoding="utf-8"
    )

    print("README 질의응답 표 갱신 완료")


# ============================================================
# 7. 메인 실행
# ============================================================

def main():

    # --------------------------------------------------------
    # 7-1. API 키 확인
    # --------------------------------------------------------

    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY가 없습니다. "
            ".env 파일을 확인하세요."
        )

    # --------------------------------------------------------
    # 7-2. 추출 및 요약 결과 복원
    # --------------------------------------------------------

    with open(
        processed / "extracted.pkl", "rb"
    ) as f:
        doc1 = pickle.load(f)

    with open(
        processed / "summaries.pkl", "rb"
    ) as f:
        doc2 = pickle.load(f)

    texts_2k_token = doc1["texts_2k_token"]
    tables = doc1["tables"]

    text_summaries = doc2["text_summaries"]
    table_summaries = doc2["table_summaries"]
    image_summaries = doc2["image_summaries"]
    img_base64_list = doc2["img_base64_list"]

    image_file_names = doc2.get(
        "image_file_names",
        [""] * len(img_base64_list)
    )

    print("결과 복원 완료")

    print(
        "텍스트 원본:",
        len(texts_2k_token)
    )
    print(
        "텍스트 요약:",
        len(text_summaries)
    )
    print("표 원본:", len(tables))
    print(
        "표 요약:",
        len(table_summaries)
    )
    print(
        "이미지 원본:",
        len(img_base64_list)
    )
    print(
        "이미지 요약:",
        len(image_summaries)
    )

    # --------------------------------------------------------
    # 7-3. 임베딩 모델 및 벡터 DB
    # --------------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        collection_name=(
            "multimodal_rag_"
            + uuid.uuid4().hex[:8]
        ),
        embedding_function=embeddings
    )

    store = InMemoryStore()

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs={"k": 8}
    )

    # 7-4. 요약과 원본 등록
    add_documents(
        retriever,
        text_summaries,
        texts_2k_token,
        kind="text"
    )

    add_documents(
        retriever,
        table_summaries,
        tables,
        kind="table"
    )

    add_documents(
        retriever,
        image_summaries,
        img_base64_list,
        kind="image",
        filenames=image_file_names
    )

    print(
        "벡터 DB 저장 완료:",
        len(vectorstore.get()["ids"])
    )

    # --------------------------------------------------------
    # 7-5. 이미지 검색 여부 확인
    # --------------------------------------------------------

    found = retriever.invoke(query)

    split_data = split_image_text_types(found)

    print(
        "검색된 원본 수:",
        len(found)
    )
    print(
        "검색된 이미지 수:",
        len(split_data["images"])
    )
    print(
        "검색된 텍스트·표 수:",
        len(split_data["texts"])
    )

    if not split_data["images"]:
        print(
            "주의: 비교 질문에서 이미지가 "
            "검색되지 않았습니다."
        )

    # --------------------------------------------------------
    # 7-6. 최종 멀티모달 체인
    # --------------------------------------------------------

    model = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        max_tokens=1024
    )

    chain = (
        {
            "context": (
                retriever
                | RunnableLambda(
                    split_image_text_types
                )
            ),
            "question": RunnablePassthrough()
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    start = time.time()

    answer = chain.invoke(query)

    print(
        f"\n답변 생성: "
        f"{time.time() - start:.1f}초"
    )
    print("질문:", query)
    print("답변:", answer)

    # --------------------------------------------------------
    # 7-7. 텍스트 전용 RAG 구성
    # --------------------------------------------------------
    # 동일한 요약 검색 → 원본 반환 구조를 사용합니다.
    # 단, 텍스트 데이터만 등록합니다.

    text_vectorstore = Chroma(
        collection_name=(
            "text_only_rag_"
            + uuid.uuid4().hex[:8]
        ),
        embedding_function=embeddings
    )

    text_store = InMemoryStore()

    text_retriever = MultiVectorRetriever(
        vectorstore=text_vectorstore,
        docstore=text_store,
        id_key=id_key,
        search_kwargs={"k": 8}
    )

    add_documents(
        text_retriever,
        text_summaries,
        texts_2k_token,
        kind="text"
    )

    # 같은 프롬프트, 같은 모델 사용
    text_only_chain = (
        {
            "context": (
                text_retriever
                | RunnableLambda(
                    split_image_text_types
                )
            ),
            "question": RunnablePassthrough()
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    # --------------------------------------------------------
    # 7-8. Before / After 비교
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Before / After 비교")
    print("질문:", query)
    print("=" * 60)

    before = text_only_chain.invoke(query)
    after = answer

    print("\n[BEFORE] 텍스트 전용 RAG")
    print(before)

    print("\n[AFTER] 멀티모달 RAG")
    print(after)

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    (results_dir / "before_after.txt").write_text(
        f"질문: {query}\n\n"
        f"[BEFORE]\n{before}\n\n"
        f"[AFTER]\n{after}\n",
        encoding="utf-8"
    )

    make_comparison_png(before, after)

    # --------------------------------------------------------
    # 7-9. 질의응답 5건
    # --------------------------------------------------------

    records = []

    for number, question in enumerate(
        QUESTIONS,
        start=1
    ):

        # 검색된 원본 목록 확인
        found = retriever.invoke(question)

        context = split_image_text_types(found)

        if question == query:
            response = after
        else:
            response = chain.invoke(question)

        unique_sources = []

        for source in context["sources"]:
            if source not in unique_sources:
                unique_sources.append(source)

        records.append({
            "number": number,
            "question": question,
            "answer": response,
            "sources": unique_sources
        })

        print(f"\n[질문 {number}]")
        print(question)
        print("[답변]")
        print(response)
        print("[검색한 원본]")
        print(unique_sources)

    # --------------------------------------------------------
    # 7-10. 최종 결과 저장
    # --------------------------------------------------------

    # JSON 결과
    (results_dir / "qa_results.json").write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # Markdown 결과
    md = [
        "# 멀티모달 RAG 질의응답 5건",
        ""
    ]

    for row in records:

        md.extend([
            f"## Q{row['number']}. {row['question']}",
            "",
            row["answer"],
            "",
            "검색된 원본: "
            + ", ".join(
                item["type"]
                for item in row["sources"]
            ),
            ""
        ])

    (results_dir / "qa_results.md").write_text(
        "\n".join(md),
        encoding="utf-8"
    )

    # README 답변 표 자동 반영
    update_readme(records)

    print("\n모든 작업 완료!")
    print("결과 저장 위치:", results_dir)

    print("qa_results.json")
    print("qa_results.md")
    print("before_after.txt")
    print("before_after.png")

    print(
        "\n주의: 실제 숫자와 그래프 해석은 "
        "원본 PDF와 대조해야 합니다."
    )


if __name__ == "__main__":
    main()
