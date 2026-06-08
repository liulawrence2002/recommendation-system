"""
Streamlit app - Goodreads Book Recommender (Project 2).

Run from this folder so imports and data paths resolve:
    cd bookrec
    streamlit run app.py

The API key is read from the environment (GEMINI_API_KEY). Without a key the
app uses a transparent heuristic re-ranker so it always runs.
"""
from __future__ import annotations

import base64
from html import escape
import os
import sys
import warnings

import streamlit as st
from dotenv import load_dotenv

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_APP_DIR, ".env"))

sys.path.insert(0, _APP_DIR)

from src import data_loader, evaluate, llm_rerank, recommend  # noqa: E402
from src.cf_model import PopularityModel  # noqa: E402

try:
    from src import cf_model

    cf_model._require_surprise()
    HAVE_SURPRISE = True
except Exception:
    HAVE_SURPRISE = False

st.set_page_config(
    page_title="BookRec",
    page_icon=":material/auto_stories:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_K = 10
DEFAULT_MIN_RATINGS = 20
RANDOM_STATE = 6604

SOURCE_REAL = "Real dataset"
SOURCE_SAMPLE = "Synthetic sample"

MODEL_LABELS = {
    "ubcf": "User-based CF",
    "ibcf": "Item-based CF",
    "baseline": "Baseline means",
    "svd": "SVD",
    "popularity": "Popularity",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&display=swap');

        :root {
            --app-bg: #f4ecdd;
            --surface: #fbf6ec;
            --surface-soft: #f0e6d4;
            --text: #2a2118;
            --muted: #8a7c66;
            --line: #e7dcc7;
            --line-strong: #d8cbb1;
            --accent: #a35421;
            --accent-soft: #f3e6d2;
            --shadow: 0 22px 60px rgba(74, 54, 32, 0.16);
            --ink: #2a2118;
            --serif: 'Fraunces', Georgia, 'Times New Roman', serif;
            --paper-edge: #e3d6bd;
            --leather: #7c4a23;
            --leather-dark: #5d3618;
        }

        html, body, .stApp, [class*="css"] {
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(1100px 520px at 50% -6%, rgba(214, 158, 86, 0.32), rgba(214, 158, 86, 0) 62%),
                radial-gradient(820px 620px at 50% 24%, rgba(255, 247, 232, 0.55), rgba(255, 247, 232, 0) 70%),
                linear-gradient(180deg, #f6efe0 0%, #efe4d0 52%, #ecdfc8 100%);
            background-attachment: fixed;
        }

        .stApp::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.5'/%3E%3C/svg%3E");
            content: "";
            inset: 0;
            mix-blend-mode: multiply;
            opacity: 0.06;
            pointer-events: none;
            position: fixed;
            z-index: 0;
        }

        .stApp::after {
            background: radial-gradient(125% 115% at 50% 30%, rgba(0, 0, 0, 0) 52%, rgba(58, 38, 18, 0.17) 100%);
            content: "";
            inset: 0;
            pointer-events: none;
            position: fixed;
            z-index: 0;
        }

        .block-container {
            position: relative;
            z-index: 1;
        }

        .block-container {
            max-width: 1160px;
            padding-top: 7rem;
            padding-bottom: 5rem;
        }

        #MainMenu,
        footer,
        header[data-testid="stHeader"],
        [data-testid="stDecoration"],
        [data-testid="stToolbar"],
        [data-testid="stHeaderActionElements"],
        .stDeployButton {
            display: none !important;
        }

        section[data-testid="stSidebar"],
        [data-testid="stSidebarNav"] {
            display: none !important;
        }

        .floating-nav {
            align-items: center;
            display: flex;
            justify-content: center;
            left: 0;
            pointer-events: none;
            position: fixed;
            right: 0;
            top: 1rem;
            z-index: 999;
        }

        .floating-nav-inner {
            align-items: center;
            backdrop-filter: blur(18px);
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(229, 229, 231, 0.96);
            border-radius: 999px;
            box-shadow: 0 18px 60px rgba(17, 17, 19, 0.10);
            display: flex;
            gap: 0.32rem;
            max-width: min(1080px, calc(100vw - 2rem));
            min-height: 4rem;
            padding: 0.46rem;
            pointer-events: auto;
            width: fit-content;
        }

        .top-brand {
            align-items: center;
            display: inline-flex;
            gap: 0.65rem;
            padding: 0.42rem 0.7rem 0.42rem 0.5rem;
            text-decoration: none !important;
        }

        .brand-mark {
            align-items: center;
            background: #111113;
            border-radius: 999px;
            color: #fff;
            display: inline-flex;
            font-size: 0.82rem;
            font-weight: 800;
            height: 2.25rem;
            justify-content: center;
            width: 2.25rem;
        }

        .brand-wordmark {
            color: var(--text);
            display: block;
            font-size: 0.95rem;
            font-weight: 790;
            line-height: 1;
        }

        .brand-caption {
            color: var(--muted);
            display: block;
            font-size: 0.68rem;
            line-height: 1.1;
            margin-top: 0.16rem;
            white-space: nowrap;
        }

        .top-links {
            align-items: center;
            display: flex;
            gap: 0.08rem;
            padding: 0 0.28rem;
        }

        .nav-pill {
            align-items: center;
            border-radius: 999px;
            color: #292a2e !important;
            display: inline-flex;
            font-size: 0.84rem;
            font-weight: 680;
            min-height: 2.55rem;
            padding: 0 0.9rem;
            text-decoration: none !important;
            white-space: nowrap;
        }

        .nav-pill:hover {
            background: #f1f0ef;
        }

        .nav-status {
            align-items: center;
            border-left: 1px solid var(--line);
            color: var(--muted);
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 680;
            gap: 0.42rem;
            min-height: 2.55rem;
            padding: 0 0.82rem;
            white-space: nowrap;
        }

        .status-dot {
            background: var(--accent);
            border-radius: 999px;
            box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
            height: 0.48rem;
            width: 0.48rem;
        }

        .nav-cta {
            align-items: center;
            background: #111113;
            border-radius: 999px;
            color: #fff !important;
            display: inline-flex;
            font-size: 0.84rem;
            font-weight: 760;
            min-height: 2.55rem;
            padding: 0 1rem;
            text-decoration: none !important;
            white-space: nowrap;
        }

        .page-kicker {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 780;
            margin-bottom: 0.72rem;
            text-transform: uppercase;
        }

        .page-title {
            font-size: 3.45rem;
            font-weight: 790;
            letter-spacing: 0;
            line-height: 0.98;
            margin: 0;
            max-width: 840px;
        }

        .page-copy {
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.65;
            margin: 1rem 0 1.8rem;
            max-width: 720px;
        }

        .section {
            border-top: 1px solid var(--line);
            margin-top: 2rem;
            padding-top: 2rem;
        }

        .section-title {
            font-size: 1.34rem;
            font-weight: 780;
            letter-spacing: 0;
            margin: 0 0 0.35rem;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.55;
            margin-bottom: 1rem;
        }

        .metric-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            min-height: 112px;
            padding: 1rem;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 760;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.7rem;
            font-weight: 790;
            line-height: 1.15;
            margin-top: 0.45rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #fff;
            border-color: var(--line) !important;
            border-radius: 8px !important;
            box-shadow: 0 8px 30px rgba(17, 17, 19, 0.04);
        }

        div[data-baseweb="select"] > div,
        input {
            border-color: var(--line-strong) !important;
            border-radius: 8px !important;
        }

        .stSlider label,
        .stSelectbox label,
        .stTextInput label {
            color: #232326;
            font-size: 0.84rem;
            font-weight: 650;
        }

        .stButton > button,
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-secondary"] {
            border-radius: 8px !important;
            font-weight: 740 !important;
            min-height: 2.75rem;
        }

        [data-testid="stBaseButton-primary"] {
            background: #111113 !important;
            border-color: #111113 !important;
            color: #fff !important;
        }

        [data-testid="stBaseButton-secondary"] {
            background: #fff !important;
            border-color: var(--line-strong) !important;
            color: #111113 !important;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.75rem 0 0.2rem;
        }

        .pill {
            align-items: center;
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: #303034;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 680;
            padding: 0.28rem 0.62rem;
        }

        .rec-list {
            display: grid;
            gap: 0.78rem;
            margin-top: 0.7rem;
        }

        .rec-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            display: grid;
            gap: 0.85rem;
            grid-template-columns: 3rem minmax(0, 1fr) auto;
            padding: 1rem;
            transition: border-color 140ms ease, box-shadow 140ms ease,
                transform 140ms ease;
        }

        .rec-card:hover {
            border-color: #cfcfd5;
            box-shadow: 0 12px 30px rgba(17, 17, 19, 0.08);
            transform: translateY(-1px);
        }

        .rank {
            align-items: center;
            background: #111113;
            border-radius: 8px;
            color: #fff;
            display: inline-flex;
            font-size: 0.86rem;
            font-weight: 790;
            height: 2.35rem;
            justify-content: center;
            width: 2.35rem;
        }

        .rec-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 750;
            line-height: 1.28;
            overflow-wrap: anywhere;
        }

        .rec-author {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.24rem;
            overflow-wrap: anywhere;
        }

        .rec-meta {
            color: var(--muted);
            display: flex;
            flex-wrap: wrap;
            font-size: 0.78rem;
            gap: 0.45rem;
            margin-top: 0.62rem;
        }

        .rec-score {
            align-self: start;
            background: var(--accent-soft);
            border: 1px solid #d6f2eb;
            border-radius: 8px;
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 780;
            padding: 0.45rem 0.62rem;
            white-space: nowrap;
        }

        .empty-state {
            align-items: center;
            background: linear-gradient(180deg, #fff 0%, #fafafa 100%);
            border: 1px dashed var(--line-strong);
            border-radius: 8px;
            color: var(--muted);
            display: flex;
            justify-content: center;
            min-height: 260px;
            padding: 1.5rem;
            text-align: center;
        }

        .empty-title {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 760;
            margin-bottom: 0.25rem;
        }

        .chat-hero {
            align-items: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: clamp(360px, 44vh, 520px);
            padding: clamp(4rem, 9vh, 6.5rem) 0 2.5rem;
            text-align: center;
        }

        .chat-title {
            font-size: clamp(2rem, 3.5vw, 3.1rem);
            font-weight: 500;
            letter-spacing: 0;
            line-height: 1.08;
            margin-bottom: 1.1rem;
        }

        .chat-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 820px;
        }

        .chat-lock {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            color: #9a3412;
            font-size: 0.94rem;
            margin: 0 auto 2.1rem;
            max-width: 860px;
            padding: 0.95rem 1.2rem;
        }

        .chat-toolbar {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
            margin: 1.4rem auto 0;
            max-width: 720px;
            text-align: center;
        }

        .st-key-chat_stage {
            margin-left: 50%;
            padding: 0 0 3.5rem;
            transform: translateX(-50%);
            width: min(1320px, calc(100vw - 6rem));
        }

        .st-key-chat_composer {
            background: #ffffff;
            border: 1px solid var(--line-strong);
            border-radius: 26px;
            box-shadow: 0 18px 60px rgba(17, 17, 19, 0.10);
            margin: 0 auto;
            max-width: 768px;
            padding: 0.55rem 0.7rem 0.6rem;
            transition: border-color 140ms ease, box-shadow 140ms ease;
        }

        .st-key-chat_composer:focus-within {
            border-color: #c4c4cb;
            box-shadow: 0 22px 72px rgba(17, 17, 19, 0.14);
        }

        .st-key-chat_composer div[data-testid="stVerticalBlock"] {
            gap: 0.15rem;
        }

        .st-key-chat_composer div[data-testid="stHorizontalBlock"] {
            align-items: center;
        }

        .st-key-chat_composer div[data-testid="stTextArea"] textarea {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            font-size: 1.04rem;
            line-height: 1.5;
            min-height: 3.4rem !important;
            padding: 0.85rem 1rem 0.45rem !important;
            resize: none;
        }

        .st-key-chat_composer div[data-baseweb="select"] > div {
            background: #f4f4f5 !important;
            border-color: #ececef !important;
            border-radius: 999px !important;
            min-height: 2.9rem;
        }

        .st-key-chat_composer .stButton > button {
            border-radius: 999px !important;
            min-height: 2.9rem;
        }

        .st-key-chat_composer div[data-testid="column"]:first-child .stButton > button {
            color: var(--muted) !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            min-width: 2.9rem;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        .st-key-chat_suggestions {
            margin: 1.1rem auto 0;
            max-width: 768px;
        }

        .st-key-chat_suggestions .stButton > button {
            background: #fff !important;
            border-color: var(--line-strong) !important;
            border-radius: 999px !important;
            box-shadow: 0 8px 24px rgba(17, 17, 19, 0.04);
            min-height: 3.45rem;
            padding-left: 1.1rem !important;
            padding-right: 1.1rem !important;
            white-space: normal;
        }

        .chat-thread {
            display: flex;
            flex-direction: column;
            gap: 1.85rem;
            margin: 0.5rem auto 2.4rem;
            max-width: 768px;
            width: 100%;
        }

        .msg {
            display: flex;
            width: 100%;
        }

        .msg-user {
            justify-content: flex-end;
        }

        .bubble-user {
            background: #f0efe9;
            border-radius: 1.4rem 1.4rem 0.4rem 1.4rem;
            color: #1f1f22;
            font-size: 0.98rem;
            line-height: 1.6;
            max-width: 82%;
            overflow-wrap: anywhere;
            padding: 0.8rem 1.15rem;
            white-space: pre-wrap;
        }

        .msg-assistant {
            align-items: flex-start;
            gap: 0.85rem;
            justify-content: flex-start;
        }

        .assistant-avatar {
            align-items: center;
            background: #111113;
            border-radius: 50%;
            color: #fff;
            display: inline-flex;
            flex: 0 0 auto;
            font-size: 0.8rem;
            font-weight: 800;
            height: 2rem;
            justify-content: center;
            margin-top: 0.15rem;
            width: 2rem;
        }

        .assistant-body {
            flex: 1 1 auto;
            min-width: 0;
        }

        .assistant-name {
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 760;
        }

        .assistant-lead {
            color: #55504b;
            font-size: 0.94rem;
            line-height: 1.6;
            margin-top: 0.2rem;
        }

        .assistant-meta {
            color: var(--muted);
            font-size: 0.76rem;
            margin-top: 0.75rem;
        }

        .thinking {
            align-items: center;
            display: inline-flex;
            gap: 0.34rem;
            margin-top: 0.6rem;
        }

        .thinking span {
            animation: thinking-bounce 1.2s infinite ease-in-out;
            background: #b9b6ad;
            border-radius: 50%;
            height: 0.5rem;
            width: 0.5rem;
        }

        .thinking span:nth-child(2) {
            animation-delay: 0.18s;
        }

        .thinking span:nth-child(3) {
            animation-delay: 0.36s;
        }

        @keyframes thinking-bounce {
            0%, 80%, 100% {
                opacity: 0.35;
                transform: translateY(0);
            }
            40% {
                opacity: 1;
                transform: translateY(-0.28rem);
            }
        }

        .pick-list {
            display: grid;
            gap: 0.6rem;
            margin-top: 0.9rem;
        }

        .pick-card {
            background: #fbfbfa;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.8rem 0.95rem;
            transition: border-color 140ms ease, box-shadow 140ms ease;
        }

        .pick-card:hover {
            border-color: #d8d8dd;
            box-shadow: 0 8px 24px rgba(17, 17, 19, 0.05);
        }

        .pick-label {
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 780;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .pick-title {
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 740;
            line-height: 1.3;
            margin-top: 0.22rem;
            overflow-wrap: anywhere;
        }

        .pick-why {
            color: #55504b;
            font-size: 0.88rem;
            line-height: 1.55;
            margin-top: 0.45rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        #filtering,
        #candidates,
        #quality,
        #personalize {
            scroll-margin-top: 6rem;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 5.8rem;
            }

            .floating-nav {
                justify-content: flex-start;
                overflow-x: auto;
                padding: 0 0.7rem;
            }

            .floating-nav-inner {
                max-width: none;
                min-width: max-content;
            }

            .top-brand {
                padding-right: 0.15rem;
            }

            .brand-wordmark,
            .brand-caption,
            .nav-status,
            .nav-cta {
                display: none;
            }

            .nav-pill {
                min-height: 2.4rem;
                padding: 0 0.68rem;
            }

            .page-title {
                font-size: 2.35rem;
            }

            .chat-hero {
                min-height: 300px;
                padding: 3rem 0 1.75rem;
            }

            .chat-subtitle {
                font-size: 0.98rem;
            }

            .st-key-chat_stage {
                margin-left: 0;
                padding-bottom: 2.5rem;
                transform: none;
                width: auto;
            }

            .st-key-chat_composer {
                border-radius: 24px;
                padding: 0.5rem;
            }

            .st-key-chat_composer div[data-testid="stHorizontalBlock"] > div:first-child {
                display: none;
            }

            .st-key-chat_composer div[data-testid="stTextArea"] textarea {
                min-height: 6rem !important;
            }

            .st-key-chat_suggestions .stButton > button {
                min-height: 3.2rem;
            }

            .rec-card {
                grid-template-columns: 2.7rem minmax(0, 1fr);
            }

            .rec-score {
                grid-column: 2;
                justify-self: start;
            }
        }
        /* ============ Warm library + cinematic layer ============ */
        .page-title,
        .chat-title,
        .section-title,
        .brand-wordmark,
        .metric-value,
        .empty-title,
        .rec-title,
        .pick-title {
            font-family: var(--serif);
        }

        .page-title,
        .chat-title {
            font-weight: 600;
            letter-spacing: -0.012em;
        }

        .section-title {
            font-weight: 600;
        }

        .floating-nav-inner {
            background: rgba(251, 246, 236, 0.86);
            border-color: rgba(216, 203, 177, 0.92);
            box-shadow: 0 18px 50px rgba(74, 54, 32, 0.18);
        }

        .brand-mark,
        .rank,
        .assistant-avatar {
            background: var(--ink);
            color: #fbf6ec;
        }

        .nav-pill {
            color: #3c3326 !important;
        }

        .nav-pill:hover {
            background: #ece1cb;
        }

        .nav-cta {
            background: var(--ink);
            color: #fbf6ec !important;
        }

        .status-dot {
            background: var(--accent);
            box-shadow: 0 0 0 4px rgba(163, 84, 33, 0.16);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            box-shadow: 0 16px 42px rgba(74, 54, 32, 0.12);
        }

        [data-testid="stBaseButton-primary"] {
            background: var(--ink) !important;
            border-color: var(--ink) !important;
            color: #fbf6ec !important;
        }

        [data-testid="stBaseButton-secondary"] {
            background: var(--surface) !important;
            border-color: var(--line-strong) !important;
            color: var(--ink) !important;
        }

        .pill {
            background: var(--surface-soft);
            color: #4a4031;
        }

        .metric-card,
        .rec-card {
            background: var(--surface);
        }

        .rec-card:hover {
            border-color: var(--line-strong);
            box-shadow: 0 18px 42px rgba(74, 54, 32, 0.16);
        }

        .rec-score {
            background: var(--accent-soft);
            border-color: #e7d2b4;
            color: var(--accent);
        }

        .empty-state {
            background: linear-gradient(180deg, var(--surface) 0%, #f3ead9 100%);
        }

        .st-key-chat_composer {
            background: var(--surface);
        }

        .st-key-chat_composer:focus-within {
            border-color: var(--accent);
            box-shadow: 0 22px 64px rgba(163, 84, 33, 0.20);
        }

        .st-key-chat_composer div[data-baseweb="select"] > div {
            background: var(--surface-soft) !important;
            border-color: var(--line) !important;
        }

        .st-key-chat_suggestions .stButton > button {
            background: var(--surface) !important;
            box-shadow: 0 10px 26px rgba(74, 54, 32, 0.10);
        }

        .bubble-user {
            background: #ece0ca;
            color: #2f261b;
        }

        .assistant-lead,
        .pick-why {
            color: #5f5340;
        }

        .thinking span {
            background: #b9a888;
        }

        .pick-card {
            background: #fdf9f0;
        }

        .pick-card:hover {
            border-color: var(--line-strong);
            box-shadow: 0 12px 28px rgba(74, 54, 32, 0.12);
        }

        .stSlider label,
        .stSelectbox label,
        .stTextInput label {
            color: #3c3326;
        }

        .chat-lock {
            background: #f6e7d0;
            border-color: #e6c79a;
            color: #8a3f12;
        }

        /* --- Cinematic hero --- */
        .cine-hero {
            align-items: center;
            display: flex;
            flex-direction: column;
            padding: 0.5rem 0 1.25rem;
            position: relative;
            text-align: center;
        }

        .cine-hero .page-kicker {
            letter-spacing: 0.14em;
            margin-top: 0.5rem;
        }

        .cine-hero .page-title,
        .cine-hero .page-copy {
            margin-left: auto;
            margin-right: auto;
        }

        /* --- Flipping book --- */
        .book {
            --w: clamp(74px, 9vw, 118px);
            --h: clamp(106px, 13vw, 170px);
            --dur: 8s;
            height: var(--h);
            perspective: 1500px;
            position: relative;
            width: calc(var(--w) * 2);
        }

        .book-hero {
            animation: book-float 6s ease-in-out infinite;
            filter: drop-shadow(0 26px 28px rgba(60, 38, 16, 0.28));
            margin: 0 auto 1.5rem;
        }

        @keyframes book-float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-9px); }
        }

        .book-3d {
            height: 100%;
            position: relative;
            transform: rotateX(13deg);
            transform-style: preserve-3d;
            width: 100%;
        }

        .book-cover {
            background: linear-gradient(145deg, var(--leather) 0%, var(--leather-dark) 100%);
            border-radius: 5px 9px 9px 5px;
            bottom: -7px;
            box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.2);
            left: -7px;
            position: absolute;
            right: -7px;
            top: -7px;
        }

        .page-static {
            background: linear-gradient(180deg, #fdf9ef 0%, #f1e6cf 100%);
            border: 1px solid var(--paper-edge);
            height: 100%;
            position: absolute;
            top: 0;
            width: var(--w);
        }

        .page-left { border-radius: 3px 0 0 3px; left: 0; }
        .page-right { border-radius: 0 3px 3px 0; right: 0; }

        .spine {
            background: linear-gradient(90deg, rgba(60, 36, 14, 0.30), rgba(60, 36, 14, 0));
            height: 100%;
            left: calc(var(--w) - 3px);
            position: absolute;
            top: 0;
            width: 7px;
            z-index: 8;
        }

        .leaf {
            animation: leaf-riffle var(--dur) ease-in-out infinite;
            background: linear-gradient(180deg, #fdfaf2 0%, #f3e9d4 100%);
            border: 1px solid var(--paper-edge);
            border-radius: 0 3px 3px 0;
            height: 100%;
            left: var(--w);
            position: absolute;
            top: 0;
            transform-origin: left center;
            width: var(--w);
        }

        .leaf::after {
            background: linear-gradient(90deg, rgba(0, 0, 0, 0) 68%, rgba(70, 44, 18, 0.12) 100%);
            content: "";
            inset: 0;
            position: absolute;
        }

        .leaf-1 { animation-delay: 0s; }
        .leaf-2 { animation-delay: 1s; }
        .leaf-3 { animation-delay: 2s; }
        .leaf-4 { animation-delay: 3s; }

        @keyframes leaf-riffle {
            0% { transform: rotateY(0deg); }
            42% { transform: rotateY(-172deg); }
            52% { transform: rotateY(-172deg); }
            94% { transform: rotateY(0deg); }
            100% { transform: rotateY(0deg); }
        }

        /* --- Ambient side books --- */
        .side-book {
            display: none;
            filter: drop-shadow(0 14px 18px rgba(60, 38, 16, 0.24));
            opacity: 0.6;
            position: fixed;
            top: 50%;
            transform: translateY(-50%);
            z-index: 1;
        }

        .book-side { --w: 46px; --h: 66px; --dur: 9s; }
        .side-left { left: 2rem; }
        .side-right { right: 2rem; }
        .side-right.book-side { --dur: 10.5s; }

        @media (min-width: 1440px) {
            .side-book { display: block; }
        }

        /* --- Image artifact panels (fixed; wide screens only) --- */
        .art-panel {
            display: none;
            height: 100vh;
            opacity: 0.9;
            pointer-events: none;
            position: fixed;
            top: 0;
            width: clamp(132px, calc((100vw - 1180px) / 2 - 6px), 360px);
            z-index: 0;
        }

        .art-panel img {
            -webkit-mask-composite: source-in;
            -webkit-mask-image:
                linear-gradient(to right, transparent 0, #000 20%, #000 80%, transparent 100%),
                linear-gradient(to bottom, transparent 0, #000 11%, #000 89%, transparent 100%);
            display: block;
            height: 100%;
            mask-composite: intersect;
            mask-image:
                linear-gradient(to right, transparent 0, #000 20%, #000 80%, transparent 100%),
                linear-gradient(to bottom, transparent 0, #000 11%, #000 89%, transparent 100%);
            object-fit: cover;
            width: 100%;
        }

        .art-left { left: 0; }
        .art-right { right: 0; }

        .art-left img {
            animation: spiral-sway 12s ease-in-out infinite;
            transform-origin: 50% 42%;
            will-change: transform;
        }

        @keyframes spiral-sway {
            0%, 100% { transform: rotate(-2deg) translateY(0) scale(1); }
            50% { transform: rotate(2deg) translateY(-12px) scale(1.03); }
        }

        .art-right img {
            animation: node-pulse 5.5s ease-in-out infinite;
            will-change: transform, filter, opacity;
        }

        @keyframes node-pulse {
            0%, 100% {
                filter: brightness(0.97) drop-shadow(0 0 0 rgba(206, 128, 66, 0));
                opacity: 0.86;
                transform: translateY(0);
            }
            50% {
                filter: brightness(1.13) drop-shadow(0 0 11px rgba(206, 128, 66, 0.5));
                opacity: 1;
                transform: translateY(-7px);
            }
        }

        @media (min-width: 1400px) {
            .art-panel { display: block; }
        }

        /* --- Image book centerpiece with flipping pages --- */
        .center-stage {
            animation: book-float 6s ease-in-out infinite;
            filter: drop-shadow(0 20px 24px rgba(60, 38, 16, 0.20));
            margin: 0 auto 1.4rem;
            position: relative;
            width: clamp(232px, 30vw, 360px);
        }

        .center-book-img {
            display: block;
            height: auto;
            width: 100%;
        }

        .flip-layer {
            bottom: 17%;
            left: 8%;
            perspective: 1700px;
            position: absolute;
            right: 8%;
            top: 16%;
            transform-style: preserve-3d;
        }

        .img-leaf {
            animation: page-flip 7s ease-in-out infinite;
            background: linear-gradient(95deg, #e6dcc6 0%, #efe8d6 18%, #ece4d2 100%);
            border: 1px solid rgba(120, 90, 50, 0.16);
            border-left-color: rgba(120, 90, 50, 0.30);
            bottom: 0;
            box-shadow: -2px 0 6px rgba(60, 40, 18, 0.12);
            left: 50%;
            position: absolute;
            top: 0;
            transform-origin: left center;
            width: 49%;
        }

        .img-leaf::before {
            border: 1px solid rgba(120, 90, 50, 0.14);
            content: "";
            inset: 7% 9%;
            position: absolute;
        }

        .leaf-a { animation-delay: 0s; }
        .leaf-b { animation-delay: 1.6s; }
        .leaf-c { animation-delay: 3.2s; }

        @keyframes page-flip {
            0% { transform: rotateY(0deg); }
            46% { transform: rotateY(-167deg); }
            54% { transform: rotateY(-167deg); }
            100% { transform: rotateY(0deg); }
        }

        @media (prefers-reduced-motion: reduce) {
            .book-hero,
            .leaf,
            .center-stage,
            .img-leaf,
            .art-left img,
            .art-right img {
                animation: none !important;
            }

            .leaf-2 { transform: rotateY(-28deg); }
            .leaf-3 { transform: rotateY(-150deg); }
            .img-leaf { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def model_label(kind: str) -> str:
    return MODEL_LABELS.get(kind, kind)


def safe_text(value, fallback: str = "") -> str:
    if value is None:
        return fallback
    try:
        if value != value:
            return fallback
    except TypeError:
        pass
    text = str(value).strip()
    return text or fallback


def format_year(value) -> str:
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return "Year unknown"
    return str(year) if year > 0 else "Year unknown"


def format_number(value) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "0"


def format_score(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def render_metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-note">{escape(note)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_recommendation_cards(recs, books) -> None:
    meta_cols = [
        "book_id",
        "original_publication_year",
        "average_rating",
        "ratings_count",
    ]
    display = recs.merge(books[meta_cols], on="book_id", how="left")

    cards = ['<div class="rec-list">']
    for rank, row in enumerate(display.itertuples(index=False), start=1):
        title = escape(safe_text(getattr(row, "title", None), "Untitled"))
        authors = escape(safe_text(getattr(row, "authors", None), "Unknown author"))
        year = escape(format_year(getattr(row, "original_publication_year", None)))
        avg = format_score(getattr(row, "average_rating", None))
        rating_count = escape(format_number(getattr(row, "ratings_count", None)))
        score = escape(format_score(getattr(row, "score", None)))
        cards.append(
            f'<article class="rec-card">'
            f'<div class="rank">{rank}</div>'
            f'<div>'
            f'<div class="rec-title">{title}</div>'
            f'<div class="rec-author">{authors}</div>'
            f'<div class="rec-meta">'
            f'<span>{year}</span>'
            f'<span>Avg {avg}</span>'
            f'<span>{rating_count} ratings</span>'
            f'</div>'
            f'</div>'
            f'<div class="rec-score">Score {score}</div>'
            f'</article>'
        )
    cards.append("</div>")
    st.markdown("\n".join(cards), unsafe_allow_html=True)


def render_assistant_message(message) -> str:
    """Build the HTML for one assistant turn (a re-ranked shortlist)."""
    picks = message.get("picks", [])
    pref = escape(safe_text(message.get("pref"), "your request"))
    lead = "Refined the shortlist" if message.get("refine") else "Here is your shortlist"
    source = escape(safe_text(message.get("source"), ""))

    parts = [
        '<div class="msg msg-assistant">',
        '<div class="assistant-avatar">B</div>',
        '<div class="assistant-body">',
        '<div class="assistant-name">BookRec</div>',
        f'<div class="assistant-lead">{lead} for &ldquo;{pref}&rdquo;.</div>',
        '<div class="pick-list">',
    ]
    if picks:
        for rank, pick in enumerate(picks, start=1):
            title = escape(safe_text(pick.title, "Untitled"))
            authors = escape(safe_text(pick.authors, "Unknown author"))
            why = escape(safe_text(pick.explanation, "Ranked for this preference."))
            parts.append(
                f'<article class="pick-card">'
                f'<div class="pick-label">Pick {rank}</div>'
                f'<div class="pick-title">{title} '
                f'<span class="rec-author">by {authors}</span></div>'
                f'<div class="pick-why">{why}</div>'
                f'</article>'
            )
    else:
        parts.append(
            '<div class="pick-why">No candidates matched closely enough. '
            'Try regenerating candidates or loosening the filters above.</div>'
        )
    parts.append("</div>")  # pick-list
    if source:
        parts.append(f'<div class="assistant-meta">Source: {source}</div>')
    parts.append("</div>")  # assistant-body
    parts.append("</div>")  # msg
    return "\n".join(parts)


def render_chat_thread(messages, pending: bool = False) -> None:
    """Render the full conversation as alternating user / assistant turns."""
    parts = ['<div class="chat-thread">']
    for message in messages:
        if message["role"] == "user":
            parts.append(
                '<div class="msg msg-user">'
                f'<div class="bubble-user">{escape(message["content"])}</div>'
                '</div>'
            )
        else:
            parts.append(render_assistant_message(message))
    if pending:
        parts.append(
            '<div class="msg msg-assistant">'
            '<div class="assistant-avatar">B</div>'
            '<div class="assistant-body">'
            '<div class="assistant-name">BookRec</div>'
            '<div class="thinking"><span></span><span></span><span></span></div>'
            '</div>'
            '</div>'
        )
    parts.append("</div>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


@st.cache_data
def load_data(source: str):
    if source == SOURCE_SAMPLE:
        return data_loader.load_sample()
    return data_loader.load("data")


@st.cache_resource
def build_model(source: str, kind: str, k: int = DEFAULT_K):
    ratings, _ = load_data(source)
    if kind == "popularity" or not HAVE_SURPRISE:
        return PopularityModel().fit(ratings)
    return cf_model.CFModel(kind=kind, k=k).fit(ratings)


@st.cache_data(show_spinner="Evaluating models on a hold-out split...")
def run_model_bakeoff(source: str, k: int, test_size: float = 0.1):
    ratings, _ = load_data(source)
    train, test = evaluate.train_test_split_ratings(
        ratings, test_size=test_size, seed=RANDOM_STATE
    )
    if not HAVE_SURPRISE:
        models = {"Popularity baseline": PopularityModel().fit(train)}
    else:
        models = {
            "Baseline": cf_model.CFModel("baseline").fit(train),
            "UBCF / pearson": cf_model.CFModel("ubcf", k=k).fit(train),
            "IBCF / cosine": cf_model.CFModel("ibcf", k=k).fit(train),
        }
    return evaluate.compare_cf_models(train, test, models), len(train), len(test)


class ScoreAdapter:
    """Wrap any model so recommend_top_n can call .score(...)."""

    content = None

    def __init__(self, model):
        self.model = model

    def score(self, user_id, user_ratings, candidate_ids):
        return self.model.predict_for_user(user_id, candidate_ids)


def compose_preference(user_turns) -> str:
    """Fold the whole conversation into one preference string for the re-ranker.

    The first turn is the base request; later turns are refinements applied in
    order. This lets a single rerank() call stay conversation-aware without the
    LLM ever inventing books outside the candidate set.
    """
    base = user_turns[0].strip()
    if len(user_turns) == 1:
        return base
    refinements = "; ".join(turn.strip() for turn in user_turns[1:])
    return (
        f"{base}. The reader then refined the request (apply in order, newest "
        f"last): {refinements}. Keep the original intent but prioritize the most "
        f"recent refinement."
    )


def submit_chat_message(prompt: str, top_k: int) -> None:
    """Append the user's turn and queue an assistant reply for the next run."""
    messages = st.session_state.setdefault("chat_messages", [])
    messages.append({"role": "user", "content": prompt})
    st.session_state["chat_pending"] = {"top_k": top_k}
    st.session_state["clear_chat_input"] = True


def resolve_pending_chat(books) -> None:
    """Run the conversation-aware RAG re-rank for the queued user turn."""
    pending = st.session_state.get("chat_pending")
    if not pending:
        return
    st.session_state["chat_pending"] = None  # clear early so we never re-enter

    messages = st.session_state.get("chat_messages", [])
    recs = st.session_state.get("cf_recs")
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    if recs is None or not user_turns:
        return

    preference = compose_preference(user_turns)
    cands = llm_rerank.candidates_from_recs(recs, books)
    picks, used_llm = llm_rerank.rerank(cands, preference, top_k=pending["top_k"])

    if used_llm:
        source = f"Gemini · {llm_rerank.DEFAULT_MODEL}"
    else:
        source = "Heuristic fallback"
        if not os.environ.get("GEMINI_API_KEY"):
            source += " (set GEMINI_API_KEY for live LLM re-ranking)"

    messages.append(
        {
            "role": "assistant",
            "picks": picks,
            "source": source,
            "used_llm": used_llm,
            "refine": len(user_turns) > 1,
            "pref": user_turns[-1],
        }
    )


inject_css()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import google.generativeai  # noqa: F401

        HAVE_GEMINI_SDK = True
    except Exception:
        HAVE_GEMINI_SDK = False

HAVE_GEMINI_KEY = bool(os.environ.get("GEMINI_API_KEY"))

if not HAVE_SURPRISE:
    nav_status = "Popularity mode"
elif HAVE_GEMINI_KEY and HAVE_GEMINI_SDK:
    nav_status = "Gemini ready"
elif HAVE_GEMINI_KEY:
    nav_status = "SDK missing"
else:
    nav_status = "Fallback mode"

st.markdown(
    '<nav class="floating-nav" aria-label="BookRec sections">'
    '<div class="floating-nav-inner">'
    '<a class="top-brand" href="#filtering">'
    '<span class="brand-mark">B</span>'
    '<span>'
    '<span class="brand-wordmark">BookRec</span>'
    '<span class="brand-caption">Personalized recommendations</span>'
    '</span>'
    '</a>'
    '<div class="top-links">'
    '<a class="nav-pill" href="#filtering">Filter</a>'
    '<a class="nav-pill" href="#candidates">Candidates</a>'
    '<a class="nav-pill" href="#quality">Audit</a>'
    '<a class="nav-pill" href="#personalize">Chat</a>'
    '</div>'
    f'<span class="nav-status"><span class="status-dot"></span>{escape(nav_status)}</span>'
    '<a class="nav-cta" href="#personalize">Personalize</a>'
    '</div>'
    '</nav>',
    unsafe_allow_html=True,
)

def book_markup(extra_classes: str, n_leaves: int = 4) -> str:
    """A CSS-only 3D book whose pages continuously riffle (fallback visual)."""
    leaves = "".join(f'<div class="leaf leaf-{i}"></div>' for i in range(1, n_leaves + 1))
    return (
        f'<div class="book {extra_classes}" aria-hidden="true">'
        '<div class="book-3d">'
        '<div class="book-cover"></div>'
        '<div class="page-static page-left"></div>'
        '<div class="page-static page-right"></div>'
        f'{leaves}'
        '<div class="spine"></div>'
        '</div>'
        '</div>'
    )


@st.cache_data
def _encode_asset(path: str, mtime: float) -> str:
    # mtime is unused in the body but is part of the cache key, so editing the
    # file on disk (new mtime) busts the cache. Do NOT prefix it with "_" —
    # st.cache_data ignores underscore-prefixed args when hashing the key.
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def asset_data_uri(filename: str):
    """Base64 data URI for an asset in ./assets, or None if it is missing.

    Streamlit can't serve local file paths inside raw HTML, so decorative art
    is embedded directly. Cached by path + mtime so regenerated files are
    picked up without restarting the server.
    """
    path = os.path.join(_APP_DIR, "assets", filename)
    if not os.path.exists(path):
        return None
    return f"data:image/webp;base64,{_encode_asset(path, os.path.getmtime(path))}"


def art_panel(uri, side: str, kind: str, fallback_classes: str) -> str:
    """Side artifact image, falling back to the CSS book if the asset is absent."""
    if uri:
        return (
            f'<div class="art-panel art-{side} art-{kind}" aria-hidden="true">'
            f'<img src="{uri}" alt=""></div>'
        )
    return book_markup(fallback_classes, n_leaves=3)


spiral_uri = asset_data_uri("spiral.webp")
nodes_uri = asset_data_uri("nodes.webp")
book_uri = asset_data_uri("book.webp")

# Ambient artifacts framing the page (fixed; wide screens only).
st.markdown(
    art_panel(spiral_uri, "left", "spiral", "side-book side-left book-side")
    + art_panel(nodes_uri, "right", "nodes", "side-book side-right book-side"),
    unsafe_allow_html=True,
)

if book_uri:
    hero_visual = (
        '<div class="center-stage" aria-hidden="true">'
        f'<img class="center-book-img" src="{book_uri}" alt="">'
        '<div class="flip-layer">'
        '<div class="img-leaf leaf-a"></div>'
        '<div class="img-leaf leaf-b"></div>'
        '<div class="img-leaf leaf-c"></div>'
        '</div>'
        '</div>'
    )
else:
    hero_visual = book_markup("book-hero")

st.markdown('<div id="filtering"></div>', unsafe_allow_html=True)
st.markdown(
    '<header class="cine-hero">'
    + hero_visual
    + '<div class="page-kicker">BookRec filtering workspace</div>'
    '<h1 class="page-title">Build the shortlist first. Personalize it at the bottom.</h1>'
    '<p class="page-copy">Select the reader, tune the collaborative filter, review the candidate books, then scroll into a chat-style personalization pass.</p>'
    '</header>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="section-title">Filtering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">These controls define the candidate pool the chat assistant will re-rank later.</div>',
        unsafe_allow_html=True,
    )

    source_col, model_col, user_col = st.columns([0.28, 0.28, 0.44])
    with source_col:
        source = st.selectbox("Data source", [SOURCE_REAL, SOURCE_SAMPLE])

    with st.spinner("Loading catalog..."):
        ratings, books = load_data(source)

    model_choices = (
        ["ubcf", "ibcf", "baseline", "svd", "popularity"]
        if HAVE_SURPRISE
        else ["popularity"]
    )
    with model_col:
        cf_kind = st.selectbox("Model", model_choices, format_func=model_label)
    with user_col:
        uid = st.selectbox(
            "Reader profile",
            sorted(ratings["user_id"].unique())[:1000],
            key="rec_user",
        )

    k_col, top_col, min_col = st.columns(3)
    with k_col:
        k_neighbors = st.slider("Neighborhood size", 5, 50, DEFAULT_K)
    with top_col:
        top_n = st.slider("Candidate count", 5, 30, 10)
    with min_col:
        min_ratings = st.slider("Minimum ratings per book", 0, 200, DEFAULT_MIN_RATINGS)

    current_config = (source, cf_kind, k_neighbors, top_n, min_ratings, uid)
    if st.session_state.get("rec_config") != current_config:
        st.session_state["cf_recs"] = None
        st.session_state["chat_messages"] = []
        st.session_state["chat_pending"] = None

    user_history = ratings[ratings["user_id"] == uid]
    st.markdown(
        f'<div class="pill-row">'
        f'<span class="pill">{escape(model_label(cf_kind))}</span>'
        f'<span class="pill">Top {top_n}</span>'
        f'<span class="pill">{len(user_history):,} reader ratings</span>'
        f'<span class="pill">Min {min_ratings:,} book ratings</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("Generate filtered candidates", type="primary", width="stretch"):
        with st.spinner("Scoring the catalog..."):
            model = build_model(source, cf_kind, k=k_neighbors)
            scorer = ScoreAdapter(model)
            st.session_state["cf_recs"] = recommend.recommend_top_n(
                uid,
                scorer,
                ratings,
                books,
                top_n=top_n,
                min_ratings=min_ratings,
                explain=False,
            )
            st.session_state["rec_config"] = current_config
            st.session_state["chat_messages"] = []
            st.session_state["chat_pending"] = None

n_users = ratings["user_id"].nunique()
n_books = len(books)
n_ratings = len(ratings)
sparsity = 1 - len(ratings) / (
    ratings["user_id"].nunique() * ratings["book_id"].nunique()
)

metrics = st.columns(4)
with metrics[0]:
    render_metric("Users", f"{n_users:,}", "reader profiles")
with metrics[1]:
    render_metric("Books", f"{n_books:,}", "catalog items")
with metrics[2]:
    render_metric("Ratings", f"{n_ratings:,}", "observed signals")
with metrics[3]:
    render_metric("Sparsity", f"{sparsity:.1%}", "matrix empty")

st.markdown('<div id="candidates" class="section"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Candidate list</div>'
    '<div class="section-copy">This is the model-produced shortlist. The chat section below reorders these books without inventing new titles.</div>',
    unsafe_allow_html=True,
)

recs = st.session_state.get("cf_recs")
if recs is None:
    with st.container(border=True):
        st.markdown(
            '<div class="empty-state"><div>'
            '<div class="empty-title">No candidates yet</div>'
            '<div>Use the filters above to generate a ranked list.</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        f'<div class="section-copy">{escape(model_label(cf_kind))} scored '
        f'{len(recs):,} books for reader {escape(str(uid))}.</div>',
        unsafe_allow_html=True,
    )
    render_recommendation_cards(recs, books)
    with st.expander("Open as table"):
        st.dataframe(
            recs[["book_id", "title", "authors", "score"]],
            width="stretch",
            hide_index=True,
        )

st.markdown('<div id="quality" class="section"></div>', unsafe_allow_html=True)
left, right = st.columns([0.34, 0.66], gap="large")
with left:
    with st.container(border=True):
        st.markdown('<div class="section-title">Quality check</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-copy">Optional 90/10 hold-out bake-off with seed {RANDOM_STATE}.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Run model audit", type="primary", width="stretch"):
            st.session_state["eval_ran"] = True
with right:
    if st.session_state.get("eval_ran"):
        results, n_train, n_test = run_model_bakeoff(source, k_neighbors)
        st.markdown(
            f'<div class="section-copy">Train: {n_train:,} ratings. Test: {n_test:,} ratings.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(results, width="stretch")
    else:
        with st.container(border=True):
            st.markdown(
                '<div class="empty-state"><div>'
                '<div class="empty-title">Audit not run</div>'
                '<div>Keep scrolling, or run the bake-off when you need model evidence.</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

st.markdown('<div id="personalize" class="section"></div>', unsafe_allow_html=True)
with st.container(key="chat_stage"):
    # Clear the composer on the run after a message is sent (Streamlit only lets
    # us reset a widget's value before the widget is instantiated).
    if st.session_state.pop("clear_chat_input", False):
        st.session_state["chat_pref"] = ""

    messages = st.session_state.setdefault("chat_messages", [])
    pending = bool(st.session_state.get("chat_pending"))
    has_recs = st.session_state.get("cf_recs") is not None
    refining = bool(messages)

    if refining:
        render_chat_thread(messages, pending=pending)
    else:
        st.markdown(
            '<div class="chat-hero">'
            '<div class="chat-title">Ready when you are.</div>'
            '<div class="chat-subtitle">Ask BookRec for a mood, genre, theme, or '
            'reading goal — then keep refining. It personalizes the filtered '
            'candidates above and never invents books outside the shortlist.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if not has_recs:
        st.markdown(
            '<div class="chat-lock">Generate filtered candidates first so the chat '
            'can re-rank real books.</div>',
            unsafe_allow_html=True,
        )

    composer_disabled = pending or not has_recs

    with st.container(key="chat_composer"):
        chat_pref = st.text_area(
            "Personalization prompt",
            key="chat_pref",
            label_visibility="collapsed",
            placeholder=(
                "Refine your shortlist — adjust tone, pace, setting, tropes to "
                "avoid, or steer it somewhere new..."
                if refining
                else "Ask for dark academia, a cozy mystery, fast-paced sci-fi, "
                "or whatever mood you're in..."
            ),
            height=92,
        )
        control_cols = st.columns(
            [0.08, 0.55, 0.19, 0.18],
            gap="small",
            vertical_alignment="center",
        )
        with control_cols[0]:
            st.button(
                "+",
                disabled=True,
                width="stretch",
                help="Attachments coming soon",
            )
        with control_cols[2]:
            chat_mode = st.selectbox(
                "Depth",
                ["Focused", "Extended"],
                label_visibility="collapsed",
                help="Focused returns 5 picks; Extended returns 8.",
            )
        with control_cols[3]:
            send = st.button(
                "Send",
                type="primary",
                disabled=composer_disabled,
                width="stretch",
            )

    personalized_k = 5 if chat_mode == "Focused" else 8

    if send and chat_pref.strip():
        submit_chat_message(chat_pref.strip(), personalized_k)
        st.rerun()

    if refining:
        refine_chips = [
            "Make them darker and moodier",
            "Lean more recent",
            "Lighter and funnier",
            "Surprise me with a wildcard",
        ]
        with st.container(key="chat_suggestions"):
            chip_cols = st.columns(len(refine_chips), gap="small")
            for col, chip in zip(chip_cols, refine_chips):
                with col:
                    if st.button(chip, disabled=composer_disabled, width="stretch"):
                        submit_chat_message(chip, personalized_k)
                        st.rerun()
        reset_cols = st.columns([0.62, 0.38])
        with reset_cols[1]:
            if st.button("↻ Start a new chat", width="stretch", disabled=pending):
                st.session_state["chat_messages"] = []
                st.session_state["chat_pending"] = None
                st.session_state["clear_chat_input"] = True
                st.rerun()
    else:
        starter_chips = [
            "Something fast-paced and adventurous",
            "A thoughtful literary list with emotional depth",
            "Dark, mysterious books with strong atmosphere",
        ]
        with st.container(key="chat_suggestions"):
            chip_cols = st.columns(len(starter_chips), gap="medium")
            for col, chip in zip(chip_cols, starter_chips):
                with col:
                    if st.button(chip, disabled=not has_recs, width="stretch"):
                        submit_chat_message(chip, personalized_k)
                        st.rerun()

    st.markdown(
        '<div class="chat-toolbar">BookRec only re-ranks the generated candidates '
        '— keep refining to steer tone, pace, or setting. It never invents books '
        'outside the shortlist.</div>',
        unsafe_allow_html=True,
    )

    # Resolve the queued turn last so the user's message + the animated "thinking"
    # dots paint first, then the re-rank runs and we rerun to show the picks.
    if st.session_state.get("chat_pending"):
        resolve_pending_chat(books)
        st.rerun()
