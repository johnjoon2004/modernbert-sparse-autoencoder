import matplotlib
from matplotlib import font_manager
from src.global_settings import settings

font_path = settings.font_path


def setup_plt_font() -> None:
    current_font = matplotlib.rcParams.get("font.family", [])
    if "NanumGothic" in current_font:
        return
    if not font_path.is_file():
        return
    font_manager.fontManager.addfont(font_path)
    font = font_manager.FontProperties(fname=font_path).get_name()
    matplotlib.rc("font", family=font)
    matplotlib.rcParams["font.family"] = "NanumGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
