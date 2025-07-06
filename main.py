import streamlit as st
from rag import process_urls, generate_answer

st.title("Real Estate Research tool")

url1 = st.sidebar.text_input("Enter URL 1")
url2 = st.sidebar.text_input("Enter URL 2")
url3 = st.sidebar.text_input("Enter URL 3")

process_url_button = st.sidebar.button("Process URLs")
placeholder = st.empty()

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url!='']
    if len(urls) == 0:
        placeholder.text("You must provide one url")

    else:
        for status in process_urls(urls):
            placeholder.text(status)
query = placeholder.text_input("Question")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)
        if sources:
            st.subheader("Sources:")
            for source in sources:
                st.write(source)
    except RuntimeError as e:
        placeholder.text(f"Error: {e}")