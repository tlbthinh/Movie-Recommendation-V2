import streamlit as st
from exploration import show_data_exploration
from recommendation import show_recommendation

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Data Exploration", "Recommendation"])

    if page == "Data Exploration":
        show_data_exploration()
    elif page == "Recommendation":
        show_recommendation()

if __name__ == "__main__":
    main()