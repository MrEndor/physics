import streamlit

from models.stone_flight import page


def main() -> None:
    pages = streamlit.navigation([
        streamlit.Page(page, title="M1 проект"),
    ])

    pages.run()


if __name__ == "__main__":
    main()
