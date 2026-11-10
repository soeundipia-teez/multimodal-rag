import os
import time
import pickle
from pathlib import Path
from collections import Counter

import pymupdf as fitz
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

pdf_dir = PROJECT_ROOT / "data"
fname = "fed_sep_2024_12.pdf"
pdf_path = pdf_dir / fname

fpath = str(pdf_dir) + os.sep

figures_dir = pdf_dir / "figures"
processed_dir = pdf_dir / "processed"


# 새 PDF의 표와 그래프가 있는 페이지 (1부터 시작)
# 벡터 그래프가 이미지로 추출되지 않을 경우를 대비
FIGURE_PAGES = {
    2: "table1_economic_projections",
    3: "figure1_projection_ranges",
    4: "figure2_dot_plot",
    5: "figure3_gdp_distribution",
    10: "figure4a_gdp_uncertainty",
    16: "table2_historical_errors",
}


# ============================================================
# 2. PDF 요소 추출 함수
# ============================================================

def extract_pdf_elements(path, fname):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.

    path: PDF 파일이 있는 폴더 경로
    fname: PDF 파일 이름
    """
    return partition_pdf(
        filename=os.path.join(path, fname),
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=os.path.join(
            path, "figures", "unstructured"
        ),
        extract_image_block_types=["Image"],
        extract_image_block_output_dir=os.path.join(
            path, "figures", "unstructured"
        ),
    )


# ============================================================
# 3. 추출한 요소 분류 함수
# ============================================================

def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출한 요소들을 테이블과 텍스트로 분류합니다.
    """
    tables = []
    texts = []

    for element in raw_pdf_elements:

        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))

        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    return texts, tables


# ============================================================
# 4. PDF 그래프 페이지 이미지 저장
# 수업 코드에 추가한 부분
# ============================================================

def save_graph_pages(pdf_file, output_dir):
    """
    PDF의 표와 그래프가 있는 페이지를 PNG로 저장합니다.

    벡터 형태의 그래프는 partition_pdf가
    이미지로 추출하지 못할 수 있기 때문에 추가했습니다.
    """
    saved = []

    with fitz.open(pdf_file) as pdf:

        for page_no, title in FIGURE_PAGES.items():

            if page_no > len(pdf):
                continue

            file_path = output_dir / f"{title}_p{page_no}.png"

            page = pdf[page_no - 1]

            pix = page.get_pixmap(
                matrix=fitz.Matrix(1.6, 1.6),
                alpha=False
            )

            pix.save(str(file_path))

            saved.append(file_path.name)

    return saved


# ============================================================
# 5. 메인 실행
# ============================================================

def main():

    print("프로젝트 경로:", PROJECT_ROOT)
    print("PDF 파일:", pdf_path)

    # PDF는 사용자가 직접 넣음
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF 파일이 없습니다: {pdf_path}\n"
            "data 폴더에 PDF를 직접 넣어주세요."
        )

    with fitz.open(pdf_path) as pdf:
        page_count = len(pdf)

    print("PDF 페이지 수:", page_count)

    if not 10 <= page_count <= 30:
        raise ValueError(
            "과제 기준은 10~30페이지입니다. PDF를 확인해주세요."
        )

    # 폴더 생성
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "unstructured").mkdir(
        parents=True, exist_ok=True
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 5-1. PDF 요소 추출
    # --------------------------------------------------------

    start = time.time()

    raw_pdf_elements = extract_pdf_elements(fpath, fname)

    print(f"추출 완료 => {time.time() - start:.1f}초")
    print("전체 요소 수:", len(raw_pdf_elements))

    # --------------------------------------------------------
    # 5-2. 요소 분류
    # --------------------------------------------------------

    texts, tables = categorize_elements(raw_pdf_elements)

    print("텍스트 블록 수:", len(texts))
    print("표 블록 수:", len(tables))

    type_counts = Counter(
        type(el).__name__ for el in raw_pdf_elements
    )

    print("원본 요소 타입:", dict(type_counts))

    if not texts:
        raise RuntimeError(
            "텍스트 추출 결과가 없습니다."
        )

    if not tables:
        print("주의: Table 객체가 추출되지 않았습니다.")

    # --------------------------------------------------------
    # 5-3. 그래프 이미지 저장
    # --------------------------------------------------------

    figure_names = save_graph_pages(
        pdf_path,
        figures_dir
    )

    print("저장한 표/그래프 이미지:", len(figure_names))

    for name in figure_names:
        print(" -", name)

    # --------------------------------------------------------
    # 5-4. 텍스트 분할
    # --------------------------------------------------------

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=200
    )

    joined_texts = " ".join(texts)

    texts_2k_token = text_splitter.split_text(joined_texts)

    print("원본 블록 수:", len(texts))
    print("분할 후 블록 수:", len(texts_2k_token))

    if texts_2k_token:
        print("\n[텍스트 청크 예시]")
        print(texts_2k_token[0][:300])

    # --------------------------------------------------------
    # 5-5. 추출 결과 저장
    # --------------------------------------------------------

    output_file = processed_dir / "extracted.pkl"

    with open(output_file, "wb") as f:
        pickle.dump(
            {
                "texts": texts,
                "tables": tables,
                "texts_2k_token": texts_2k_token,
                "figure_names": figure_names,
            },
            f
        )

    print("\n저장 완료:", output_file)
    print("다음 실행: python src/summarize.py")


if __name__ == "__main__":
    main()
