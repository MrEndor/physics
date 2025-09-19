import streamlit


def main() -> None:
    pages = streamlit.navigation([
        streamlit.Page("./models/stone_flight/__init__.py", title="M1 проект")
    ])

    pages.run()


if __name__ == "__main__":
    main()
