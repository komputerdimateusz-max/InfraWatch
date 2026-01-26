# =========================================================
# FILE: src/infrawatch/ui/app.py
# =========================================================
import streamlit as st
from infrawatch.config import load_config
from infrawatch.logging_conf import setup_logging

def main() -> None:
    cfg = load_config()
    setup_logging(cfg.log_level)

    st.set_page_config(page_title="InfraWatch MVP", layout="wide")
    st.title("InfraWatch™ – MVP")
    st.caption("Vegetation risk monitoring near traction & power networks")

    st.success("Skeleton application is running.")
    st.write("Data directory:", cfg.data_dir)

if __name__ == "__main__":
    main()
