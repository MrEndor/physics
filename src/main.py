import streamlit

from models.billiard import page as billiard_page
from models.rolling_ball import page as rolling_ball_page
from models.stone_flight import page as stone_flight_page


def main() -> None:
    pages = streamlit.navigation([
        streamlit.Page(
            stone_flight_page, title="M1 проект", url_path="stone_flight"
        ),
        streamlit.Page(billiard_page, title="M2 Бильярд", url_path="billiard"),
        streamlit.Page(
            rolling_ball_page, title="M4 Шар на столе", url_path="rolling_ball"
        ),
    ])

    pages.run()


if __name__ == "__main__":
    main()
