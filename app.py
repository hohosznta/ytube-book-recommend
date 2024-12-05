import streamlit as st
from phi.tools.youtube_tools import YouTubeTools
from PIL import Image
import requests
from io import BytesIO

from assistant import get_chunk_summarizer, get_video_summarizer, query_qdrant  # type: ignore

st.set_page_config(
    page_title="YouREAD",
    page_icon=":books:",
)
st.title("YouREAD:books:  새로운 경험의 창조")
st.subheader("영상에서 책으로, 지식을 한 걸음 더 나아가세요.")


def main() -> None:
    # Get model
    llm_model =  "bnksys/yanolja-eeve-korean-instruct-10.8b"
    # Get video url
    video_url = st.sidebar.text_input(":video_camera: Video URL")
    # Button to generate report
    generate_report = st.sidebar.button("책 추천해드립니다.")
    if generate_report:
        st.session_state["youtube_url"] = video_url

    if "youtube_url" in st.session_state:
        _url = st.session_state["youtube_url"]
        youtube_tools = YouTubeTools(languages=["ko", "en"])
        video_captions = None
        video_summarizer = get_video_summarizer(model=llm_model)

        with st.status("Parsing Video", expanded=False) as status:
            with st.container():
                video_container = st.empty()
                video_container.video(_url)

            video_data = youtube_tools.get_youtube_video_data(_url)
            with st.container():
                video_data_container = st.empty()
                video_data_container.json(video_data)
            status.update(label="Video", state="complete", expanded=False)

        with st.status("Reading Captions", expanded=False) as status:
            video_captions = youtube_tools.get_youtube_video_captions(_url)
            with st.container():
                video_captions_container = st.empty()
                video_captions_container.write(video_captions)
            status.update(label="Captions processed", state="complete", expanded=False)

        if not video_captions:
            st.write("Sorry could not parse video. Please try again or use a different video.")
            return

        chunks = []
        num_chunks = 0
        words = video_captions.split()
        for i in range(0, len(words), 4000):
            num_chunks += 1
            chunks.append(" ".join(words[i : (i + 4000)]))

        if num_chunks > 1:
            chunk_summaries = []
            for i in range(num_chunks):
                with st.status(f"Summarizing chunk: {i+1}", expanded=False) as status:
                    chunk_summary = ""
                    chunk_container = st.empty()
                    chunk_summarizer = get_chunk_summarizer(model=llm_model)
                    chunk_info = f"Video data: {video_data}\n\n"
                    chunk_info += f"{chunks[i]}\n\n"
                    for delta in chunk_summarizer.run(chunk_info):
                        chunk_summary += delta  # type: ignore
                        chunk_container.markdown(chunk_summary)
                    chunk_summaries.append(chunk_summary)
                    status.update(label=f"Chunk {i+1} summarized", state="complete", expanded=False)

            with st.spinner("Generating Summary"):
                summary = ""
                summary_container = st.empty()
                video_info = f"Video URL: {_url}\n\n"
                video_info += f"Video Data: {video_data}\n\n"
                video_info += "Summaries:\n\n"
                for i, chunk_summary in enumerate(chunk_summaries, start=1):
                    video_info += f"Chunk {i}:\n\n{chunk_summary}\n\n"
                    video_info += "---\n\n"

                for delta in video_summarizer.run(video_info):
                    summary += delta  # type: ignore
                    # summary_container.markdown(summary)
        else:
            with st.spinner("Generating Summary"):
                summary = ""
                # summary_container = st.empty()
                video_info = f"Video URL: {_url}\n\n"
                video_info += f"Video Data: {video_data}\n\n"
                video_info += f"Captions: {video_captions}\n\n"

                for delta in video_summarizer.run(video_info):
                    summary += delta  # type: ignore
                    # summary_container.markdown(summary)
        
        query_results= query_qdrant(summary, 'Books')
        for i, article in enumerate(query_results):
            st.header(f"{i + 1}. {article.payload['title']}:open_book:")
            response = requests.get(article.payload['url'])
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='책 사진')
            st.markdown(f"작가: {article.payload['author']}")
            st.markdown("""<hr style="height:1px;border:none;color:#FF4433;background-color:#FF4433;" /> """, unsafe_allow_html=True) 


    else:
        st.write("유튜브 링크를 입력해주세요. ")

    st.sidebar.markdown("---")
    if st.sidebar.button("Restart"):
        st.rerun()


main()
