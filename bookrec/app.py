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
from dataclasses import asdict
from html import escape
import os
import re
import sys
import urllib.parse
import warnings

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_APP_DIR, ".env"))

sys.path.insert(0, _APP_DIR)

from src import data_loader, evaluate, llm_rerank, rag_pipeline, recommend  # noqa: E402
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
DEFAULT_UBCF_K = 21
DEFAULT_MIN_RATINGS = 20
RANDOM_STATE = 6604

SOURCE_REAL = "Real dataset"
SOURCE_SAMPLE = "Synthetic sample"

NOTEBOOK_METRICS = [
    ("Baseline", 0.6568, 0.7912, 0.7178),
    ("UBCF (pearson)", 0.6586, 0.7930, 0.7196),
    ("IBCF (cosine)", 0.6414, 0.7734, 0.7012),
]

MODEL_LABELS = {
    "ubcf": "User-based CF (best)",
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
            align-items: center;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            display: grid;
            gap: 0.8rem;
            grid-template-columns: 2.35rem 2.5rem minmax(0, 1fr) auto;
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

        .rec-cover {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(74, 54, 32, 0.18);
            display: block;
            height: 3.6rem;
            object-fit: cover;
            transition: box-shadow 160ms ease, transform 160ms ease;
            width: 2.5rem;
        }

        .rec-cover--empty {
            align-items: center;
            color: var(--muted);
            display: flex;
            font-size: 1.05rem;
            justify-content: center;
        }

        .rec-cover-link {
            display: block;
            line-height: 0;
        }

        .rec-cover-link:hover .rec-cover {
            box-shadow: 0 7px 18px rgba(74, 54, 32, 0.30);
            transform: translateY(-1px);
        }

        .rec-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 750;
            line-height: 1.28;
            overflow-wrap: anywhere;
        }

        .rec-title-link {
            color: inherit;
            text-decoration: none;
        }

        .rec-title-link:hover {
            text-decoration: underline;
            text-decoration-color: var(--accent);
            text-underline-offset: 2px;
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
            gap: 2.4rem;
            margin: 0.5rem auto 2.6rem;
            max-width: 720px;
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
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 1.35rem 1.35rem 0.45rem 1.35rem;
            color: #2f261b;
            font-size: 0.97rem;
            line-height: 1.6;
            max-width: 78%;
            overflow-wrap: anywhere;
            padding: 0.7rem 1.1rem;
            white-space: pre-wrap;
        }

        .msg-assistant {
            align-items: flex-start;
            gap: 0.95rem;
            justify-content: flex-start;
        }

        .assistant-avatar {
            align-items: center;
            background: var(--ink);
            border-radius: 50%;
            color: #fbf6ec;
            display: inline-flex;
            flex: 0 0 auto;
            font-family: var(--serif);
            font-size: 0.82rem;
            font-weight: 600;
            height: 1.95rem;
            justify-content: center;
            margin-top: 0.1rem;
            width: 1.95rem;
        }

        .assistant-body {
            flex: 1 1 auto;
            min-width: 0;
        }

        .assistant-name {
            color: var(--text);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }

        .assistant-lead {
            color: #4a4031;
            font-size: 1.04rem;
            line-height: 1.62;
            margin-top: 0.35rem;
        }

        .assistant-meta {
            color: var(--muted);
            font-size: 0.74rem;
            margin-top: 0.9rem;
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
            gap: 0.85rem;
            margin-top: 1.25rem;
        }

        .pick-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            display: grid;
            gap: 0.05rem 0.85rem;
            grid-template-columns: 2.4rem 2.8rem minmax(0, 1fr);
            padding: 1.05rem 1.15rem;
            transition: border-color 160ms ease, box-shadow 160ms ease,
                transform 160ms ease;
        }

        .pick-card:hover {
            border-color: var(--line-strong);
            box-shadow: 0 14px 32px rgba(74, 54, 32, 0.12);
            transform: translateY(-1px);
        }

        .pick-rank {
            align-items: center;
            background: var(--ink);
            border-radius: 50%;
            color: #fbf6ec;
            display: inline-flex;
            font-family: var(--serif);
            font-size: 0.92rem;
            font-weight: 600;
            grid-column: 1;
            grid-row: 1 / span 3;
            height: 2.4rem;
            justify-content: center;
            margin-top: 0.05rem;
            width: 2.4rem;
        }

        .pick-cover {
            align-self: start;
            display: block;
            grid-column: 2;
            grid-row: 1 / span 3;
            margin-top: 0.12rem;
        }

        .pick-cover-img {
            border-radius: 5px;
            box-shadow: 0 3px 10px rgba(74, 54, 32, 0.22);
            display: block;
            height: 4.2rem;
            object-fit: cover;
            transition: box-shadow 160ms ease, transform 160ms ease;
            width: 2.8rem;
        }

        .pick-cover:hover .pick-cover-img {
            box-shadow: 0 7px 18px rgba(74, 54, 32, 0.30);
            transform: translateY(-1px);
        }

        .pick-cover--empty {
            align-items: center;
            background: var(--surface-soft);
            border: 1px solid var(--line);
            box-shadow: none;
            color: var(--muted);
            display: flex;
            font-size: 1.1rem;
            justify-content: center;
        }

        .pick-title {
            color: var(--text);
            font-size: 1.04rem;
            font-weight: 600;
            grid-column: 3;
            line-height: 1.3;
            overflow-wrap: anywhere;
        }

        .pick-title-link {
            color: inherit;
            text-decoration: none;
        }

        .pick-title-link:hover {
            text-decoration: underline;
            text-decoration-color: var(--accent);
            text-underline-offset: 2px;
        }

        .pick-author {
            color: var(--muted);
            font-size: 0.85rem;
            font-weight: 500;
        }

        .pick-desc {
            color: #4a4031;
            font-size: 0.92rem;
            grid-column: 3;
            line-height: 1.58;
            margin-top: 0.5rem;
        }

        .pick-why {
            border-left: 2px solid var(--accent);
            color: #5f5340;
            font-size: 0.9rem;
            grid-column: 3;
            line-height: 1.55;
            margin-top: 0.7rem;
            padding-left: 0.8rem;
        }

        .pick-why-label {
            color: var(--accent);
            display: block;
            font-size: 0.68rem;
            font-weight: 780;
            letter-spacing: 0.06em;
            margin-bottom: 0.2rem;
            text-transform: uppercase;
        }

        .pick-empty {
            color: #5f5340;
            font-size: 0.92rem;
            line-height: 1.55;
            margin-top: 0.6rem;
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
                grid-template-columns: 2.1rem 2.3rem minmax(0, 1fr);
            }

            .rec-score {
                grid-column: 3;
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

        .thinking span {
            background: #b9a888;
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

        /* ============ Sequential DAG: intent chips + reasoning trace ============ */
        .intent-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.65rem 0 0.2rem;
        }

        .intent-chip {
            align-items: center;
            background: var(--accent-soft);
            border: 1px solid #e7d2b4;
            border-radius: 999px;
            color: var(--accent);
            display: inline-flex;
            font-size: 0.72rem;
            font-weight: 760;
            letter-spacing: 0.03em;
            padding: 0.24rem 0.6rem;
            text-transform: uppercase;
        }

        .intent-chip--avoid {
            background: var(--surface-soft);
            border-color: var(--line-strong);
            color: var(--muted);
            text-decoration: line-through;
            text-decoration-thickness: 1px;
        }

        .reasoning-trace {
            background: linear-gradient(180deg, #fefcf6 0%, #fbf6ec 100%);
            border: 1px solid var(--line);
            border-radius: 14px;
            margin-top: 1rem;
            overflow: hidden;
            transition: border-color 160ms ease, box-shadow 160ms ease;
        }

        .reasoning-trace[open] {
            border-color: var(--line-strong);
            box-shadow: 0 16px 38px rgba(74, 54, 32, 0.12);
        }

        .reasoning-trace > summary {
            align-items: center;
            color: var(--ink);
            cursor: pointer;
            display: flex;
            font-size: 0.86rem;
            font-weight: 680;
            gap: 0.5rem;
            letter-spacing: 0.01em;
            list-style: none;
            padding: 0.85rem 1.05rem;
            user-select: none;
        }

        .reasoning-trace > summary:hover {
            background: rgba(163, 84, 33, 0.05);
        }

        .reasoning-trace > summary::-webkit-details-marker {
            display: none;
        }

        .reasoning-trace > summary::before {
            color: var(--accent);
            content: "›";
            display: inline-block;
            font-size: 1.1rem;
            font-weight: 800;
            transition: transform 160ms ease;
        }

        .reasoning-trace[open] > summary::before {
            transform: rotate(90deg);
        }

        .trace-intro {
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.5;
            padding: 0.7rem 1.05rem 0.2rem;
        }

        .trace-stages {
            counter-reset: trace-step;
            padding: 0.4rem 1.05rem 0.9rem;
            position: relative;
        }

        /* vertical connector running through the numbered steps */
        .trace-stages::before {
            background: var(--line-strong);
            bottom: 1.7rem;
            content: "";
            left: calc(1.05rem + 0.86rem);
            position: absolute;
            top: 1.4rem;
            width: 2px;
            z-index: 0;
        }

        .trace-stage {
            padding: 0.7rem 0 0.7rem 2.55rem;
            position: relative;
            z-index: 1;
        }

        .trace-stage-head {
            align-items: center;
            display: flex;
            gap: 0.55rem;
        }

        .trace-step-no {
            align-items: center;
            background: var(--surface);
            border: 2px solid var(--line-strong);
            border-radius: 50%;
            color: var(--accent);
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 780;
            height: 1.72rem;
            justify-content: center;
            left: 0;
            position: absolute;
            top: 0.62rem;
            width: 1.72rem;
        }

        .trace-stage-name {
            color: var(--text);
            flex: 1 1 auto;
            font-family: var(--serif);
            font-size: 1.02rem;
            font-weight: 600;
        }

        .stage-badge {
            align-items: center;
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.66rem;
            font-weight: 740;
            gap: 0.34rem;
            letter-spacing: 0.03em;
            padding: 0.22rem 0.58rem;
            text-transform: uppercase;
        }

        .stage-badge .status-dot {
            box-shadow: none;
        }

        .stage-badge--live {
            background: var(--accent-soft);
            color: var(--accent);
        }

        .stage-badge--live .status-dot {
            background: var(--accent);
        }

        .stage-badge--heuristic {
            background: var(--surface-soft);
            color: var(--muted);
        }

        .stage-badge--heuristic .status-dot {
            background: var(--muted);
        }

        .trace-note {
            color: #6b5d48;
            font-size: 0.84rem;
            line-height: 1.5;
            margin-top: 0.35rem;
        }

        .trace-body {
            color: #5f5340;
            font-size: 0.84rem;
            line-height: 1.55;
            margin-top: 0.6rem;
        }

        .trace-line {
            overflow-wrap: anywhere;
            padding: 0.16rem 0;
        }

        .trace-head {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
            text-transform: uppercase;
        }

        .trace-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.34rem;
            margin-top: 0.1rem;
        }

        .trace-chip {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: #4a4031;
            font-size: 0.74rem;
            font-weight: 620;
            padding: 0.18rem 0.55rem;
        }

        .trace-chip--avoid {
            color: var(--muted);
            text-decoration: line-through;
            text-decoration-thickness: 1px;
        }

        .trace-strength {
            background: var(--accent-soft);
            border-radius: 6px;
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 740;
            padding: 0.1rem 0.4rem;
            white-space: nowrap;
        }

        .trace-rank {
            background: var(--ink);
            border-radius: 5px;
            color: #fbf6ec;
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 760;
            min-width: 1.1rem;
            padding: 0.04rem 0.3rem;
            text-align: center;
        }

        .trace-book {
            color: var(--text);
            font-weight: 680;
        }

        .trace-flag {
            color: var(--muted);
            font-size: 0.76rem;
            font-style: italic;
        }

        .clarify-hint {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.5;
            margin-top: 0.7rem;
        }

        /* --- pending stage-progress strip --- */
        .stage-progress {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.7rem 0 0.2rem;
        }

        .stage-pill {
            align-items: center;
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: #4a4031;
            display: inline-flex;
            font-size: 0.74rem;
            font-weight: 720;
            gap: 0.4rem;
            padding: 0.26rem 0.62rem;
        }

        .stage-pill-dot {
            animation: thinking-bounce 1.2s infinite ease-in-out;
            background: var(--accent);
            border-radius: 50%;
            height: 0.42rem;
            width: 0.42rem;
        }

        .stage-pill:nth-child(2) .stage-pill-dot {
            animation-delay: 0.18s;
        }

        .stage-pill:nth-child(3) .stage-pill-dot {
            animation-delay: 0.36s;
        }

        @media (prefers-reduced-motion: reduce) {
            .book-hero,
            .leaf,
            .center-stage,
            .img-leaf,
            .art-left img,
            .art-right img,
            .stage-pill-dot {
                animation: none !important;
            }

            .leaf-2 { transform: rotateY(-28deg); }
            .leaf-3 { transform: rotateY(-150deg); }
            .img-leaf { display: none; }
        }

        /* ===================== Premium chat polish ===================== */
        /* Slightly airier thread + crisper reading rhythm. */
        .chat-thread {
            gap: 2.1rem;
        }

        .chat-title {
            letter-spacing: -0.015em;
        }

        .chat-subtitle {
            color: #6b5d48;
        }

        /* Assistant identity — a warm gradient medallion with a soft ring. */
        .assistant-avatar {
            background: linear-gradient(150deg, #6b4423 0%, var(--ink) 100%);
            box-shadow:
                0 4px 12px rgba(42, 33, 24, 0.26),
                0 0 0 3px rgba(163, 84, 33, 0.10);
            font-family: var(--serif);
            letter-spacing: 0.01em;
        }

        .assistant-name {
            letter-spacing: 0.01em;
        }

        /* User turn — a soft, layered paper bubble. */
        .bubble-user {
            background: linear-gradient(180deg, #f1e6d1 0%, #ece0ca 100%);
            border: 1px solid var(--paper-edge);
            border-radius: 1.5rem 1.5rem 0.5rem 1.5rem;
            box-shadow: 0 2px 8px rgba(74, 54, 32, 0.08);
        }

        /* Pick cards — layered depth, soft inner highlight, lifted hover. */
        .pick-card {
            background: linear-gradient(180deg, #fdf9f0 0%, var(--surface) 100%);
            border-color: var(--line);
            border-radius: 16px;
            box-shadow:
                0 1px 0 rgba(255, 255, 255, 0.6) inset,
                0 6px 18px rgba(74, 54, 32, 0.07);
            padding: 1.15rem 1.25rem;
            transition: border-color 180ms ease, box-shadow 180ms ease,
                transform 180ms ease;
        }

        .pick-card:hover {
            border-color: var(--line-strong);
            box-shadow:
                0 1px 0 rgba(255, 255, 255, 0.6) inset,
                0 18px 40px rgba(74, 54, 32, 0.16);
            transform: translateY(-2px);
        }

        .pick-rank {
            background: linear-gradient(150deg, #6b4423 0%, var(--ink) 100%);
            box-shadow:
                0 4px 12px rgba(42, 33, 24, 0.24),
                0 0 0 3px rgba(163, 84, 33, 0.08);
        }

        /* The "why it ranks here" line — a tinted, inset rationale chip. */
        .pick-why {
            background: linear-gradient(180deg, #fbeedd 0%, var(--accent-soft) 100%);
            border-left: 2px solid var(--accent);
            border-radius: 0 9px 9px 0;
            margin-top: 0.8rem;
            padding: 0.55rem 0.75rem;
        }

        .pick-why-label {
            letter-spacing: 0.07em;
        }

        /* Reasoning disclosure — quieter, rounder, premium. */
        .reasoning-trace {
            border-radius: 16px;
        }

        .reasoning-trace > summary {
            font-weight: 640;
            letter-spacing: 0.01em;
        }

        /* Composer — gradient surface, layered shadow, accent focus ring. */
        .st-key-chat_composer {
            background: linear-gradient(180deg, #fffdf8 0%, var(--surface) 100%);
            border-color: var(--line);
            border-radius: 28px;
            box-shadow:
                0 1px 0 rgba(255, 255, 255, 0.7) inset,
                0 2px 6px rgba(74, 54, 32, 0.06),
                0 24px 60px rgba(74, 54, 32, 0.12);
            padding: 0.7rem 0.85rem 0.75rem;
        }

        .st-key-chat_composer:focus-within {
            border-color: var(--accent);
            box-shadow:
                0 0 0 3px rgba(163, 84, 33, 0.12),
                0 26px 72px rgba(163, 84, 33, 0.18);
        }

        .st-key-chat_composer div[data-testid="stTextArea"] textarea::placeholder {
            color: #ab9d85;
        }

        .st-key-chat_composer div[data-baseweb="select"] > div {
            background: var(--surface-soft) !important;
            border-color: var(--line) !important;
            box-shadow: none !important;
        }

        /* Send — circular premium primary with a confident lift. */
        .st-key-chat_composer div[data-testid="column"]:last-child .stButton > button {
            background: linear-gradient(180deg, #3a2c1c 0%, var(--ink) 100%) !important;
            border: 0 !important;
            border-radius: 999px !important;
            box-shadow: 0 6px 18px rgba(42, 33, 24, 0.26);
            color: #fbf6ec !important;
            font-weight: 760 !important;
            letter-spacing: 0.02em;
            transition: transform 150ms ease, box-shadow 150ms ease,
                filter 150ms ease;
        }

        .st-key-chat_composer div[data-testid="column"]:last-child
            .stButton > button:hover:not(:disabled) {
            box-shadow: 0 10px 28px rgba(42, 33, 24, 0.34);
            filter: brightness(1.07);
            transform: translateY(-1px);
        }

        .st-key-chat_composer div[data-testid="column"]:last-child
            .stButton > button:disabled {
            box-shadow: none;
            opacity: 0.45;
        }

        /* Suggestion chips — refined, responsive hover. */
        .st-key-chat_suggestions .stButton > button {
            transition: transform 150ms ease, box-shadow 150ms ease,
                border-color 150ms ease;
        }

        .st-key-chat_suggestions .stButton > button:hover {
            border-color: var(--accent) !important;
            box-shadow: 0 12px 30px rgba(74, 54, 32, 0.14) !important;
            transform: translateY(-1px);
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
    # Pull the cover-image link straight from the catalog when present (the
    # assignment dataset ships Goodreads `small_image_url`/`image_url`; the
    # synthetic sample has neither, so we fall back to a placeholder tile).
    for col in ("small_image_url", "image_url"):
        if col in books.columns and col not in meta_cols:
            meta_cols.append(col)
    display = recs.merge(books[meta_cols], on="book_id", how="left")

    cards = ['<div class="rec-list">']
    for rank, row in enumerate(display.itertuples(index=False), start=1):
        title_raw = safe_text(getattr(row, "title", None), "Untitled")
        authors_raw = safe_text(getattr(row, "authors", None), "Unknown author")
        title = escape(title_raw)
        authors = escape(authors_raw)
        year = escape(format_year(getattr(row, "original_publication_year", None)))
        avg = format_score(getattr(row, "average_rating", None))
        rating_count = escape(format_number(getattr(row, "ratings_count", None)))
        score = escape(format_score(getattr(row, "score", None)))
        cover = safe_text(getattr(row, "small_image_url", None)) or safe_text(
            getattr(row, "image_url", None)
        )
        # Same cover + Goodreads link treatment as the chat pick cards.
        link = escape(_goodreads_url(cover, title_raw, authors_raw))
        if cover.startswith("http"):
            cover_inner = (
                f'<img class="rec-cover" src="{escape(cover)}" alt="" '
                f'loading="lazy" referrerpolicy="no-referrer">'
            )
        else:
            cover_inner = (
                '<span class="rec-cover rec-cover--empty" aria-hidden="true">📖</span>'
            )
        cards.append(
            f'<article class="rec-card">'
            f'<div class="rank">{rank}</div>'
            f'<a class="rec-cover-link" href="{link}" target="_blank" '
            f'rel="noopener noreferrer">{cover_inner}</a>'
            f'<div>'
            f'<div class="rec-title">'
            f'<a class="rec-title-link" href="{link}" target="_blank" '
            f'rel="noopener noreferrer">{title}</a></div>'
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


def render_intent_chips(intent) -> str:
    """A row of pills summarizing the DAG's extracted intent (Stage A).

    `intent` is the asdict() of rag_pipeline.Intent. Returns "" when there is
    nothing meaningful to show so old messages stay clean.
    """
    if not intent:
        return ""
    chips = []
    mood = safe_text(intent.get("mood"))
    if mood:
        chips.append(f'<span class="intent-chip">{escape(mood)}</span>')
    pace = safe_text(intent.get("pace"))
    if pace and pace.lower() != "any":
        chips.append(f'<span class="intent-chip">{escape(pace)} pace</span>')
    recency = safe_text(intent.get("recency"))
    if recency and recency.lower() != "any":
        chips.append(f'<span class="intent-chip">{escape(recency)}</span>')
    for genre in (intent.get("genres") or [])[:3]:
        label = safe_text(genre)
        if label:
            chips.append(f'<span class="intent-chip">{escape(label)}</span>')
    for theme in (intent.get("themes") or [])[:2]:
        label = safe_text(theme)
        if label:
            chips.append(f'<span class="intent-chip">{escape(label)}</span>')
    for avoid in (intent.get("avoid") or [])[:3]:
        label = safe_text(avoid)
        if label:
            chips.append(
                f'<span class="intent-chip intent-chip--avoid">no {escape(label)}</span>'
            )
    if not chips:
        return ""
    return '<div class="intent-row">' + "".join(chips) + "</div>"


def _match_strength(value) -> str:
    """Turn a 0..1 relevance score into a plain-language match label."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    if score >= 0.66:
        return "Strong match"
    if score >= 0.33:
        return "Fair match"
    return "Light match"


def _chip_line(label: str) -> str:
    return f'<span class="trace-chip">{escape(label)}</span>'


def _trace_output_summary(name: str, output: dict) -> str:
    """Plain-language view of one step's result, written for a general reader."""
    if not output:
        return ""
    if name == "Understand your request":
        chips = []
        mood = safe_text(output.get("mood"))
        if mood:
            chips.append(_chip_line(mood))
        for key in ("pace", "recency"):
            val = safe_text(output.get(key))
            if val and val.lower() != "any":
                chips.append(_chip_line(f"{val} {key}" if key == "pace" else val))
        for genre in (output.get("genres") or []):
            label = safe_text(genre)
            if label:
                chips.append(_chip_line(label))
        for theme in (output.get("themes") or [])[:3]:
            label = safe_text(theme)
            if label:
                chips.append(_chip_line(label))
        for avoid in (output.get("avoid") or []):
            label = safe_text(avoid)
            if label:
                chips.append(
                    f'<span class="trace-chip trace-chip--avoid">avoid {escape(label)}</span>'
                )
        if not chips:
            return '<div class="trace-line">Nothing specific yet — we’ll ask a quick question.</div>'
        return '<div class="trace-chips">' + "".join(chips) + "</div>"
    if name == "Check we have enough":
        if output.get("question"):
            q = safe_text(output.get("question"))
            opts = [safe_text(o) for o in (output.get("options") or []) if safe_text(o)]
            rows = [f'<div class="trace-line">We asked: <em>“{escape(q)}”</em></div>']
            if opts:
                rows.append(
                    '<div class="trace-chips">'
                    + "".join(_chip_line(o) for o in opts)
                    + "</div>"
                )
            return "".join(rows)
        return '<div class="trace-line">Enough detail to go on — moving to the books.</div>'
    if name == "Rate every book":
        rows = []
        for s in output.get("scored", [])[:6]:
            title = safe_text(s.get("title")) or f"Book #{safe_text(s.get('book_id'))}"
            strength = _match_strength(s.get("relevance"))
            reason = safe_text(s.get("reason"))
            flags = [safe_text(f) for f in (s.get("flags") or []) if safe_text(f)]
            line = (
                f'<span class="trace-strength">{escape(strength)}</span> '
                f'<span class="trace-book">{escape(title)}</span>'
            )
            if reason:
                line += f' — {escape(reason)}'
            if flags:
                line += ' <span class="trace-flag">set aside: '
                line += escape(", ".join(f.replace("avoid:", "") for f in flags))
                line += "</span>"
            rows.append(f'<div class="trace-line">{line}</div>')
        head = (
            f'We rated {output.get("count", len(rows))} books — '
            f'showing the {len(rows)} strongest:'
        )
        return f'<div class="trace-line trace-head">{escape(head)}</div>' + "".join(rows)
    if name == "Pick the final list":
        rows = []
        for i, p in enumerate(output.get("picks", []), start=1):
            title = safe_text(p.get("title"), "Untitled")
            desc = safe_text(p.get("description"))
            why = safe_text(p.get("explanation"))
            line = (
                f'<span class="trace-rank">{i}</span> '
                f'<span class="trace-book">{escape(title)}</span>'
            )
            if desc:
                line += f' — {escape(desc)}'
            if why:
                line += f' <span class="trace-flag">Why: {escape(why)}</span>'
            rows.append(f'<div class="trace-line">{line}</div>')
        return "".join(rows)
    return ""


# Map the pipeline's internal stage names to reader-friendly step titles.
TRACE_STEP_NAMES = {
    "Intent": "Understand your request",
    "Clarify": "Check we have enough",
    "Scoring": "Rate every book",
    "Re-rank": "Pick the final list",
}


def render_reasoning_trace(trace) -> str:
    """A collapsible step-by-step explanation of how the shortlist was built.

    `trace` is a list of asdict() rag_pipeline.StageTrace dicts. Native HTML
    <details>, so it works inside st.markdown with no JS. Returns "" when absent.
    Written for a non-technical reader: numbered steps, plain notes, and an
    "AI" vs "Smart rules" badge instead of LLM jargon.
    """
    if not trace:
        return ""
    stages = []
    for i, stage in enumerate(trace, start=1):
        raw_name = safe_text(stage.get("name"), "Stage")
        name = TRACE_STEP_NAMES.get(raw_name, raw_name)
        used_llm = bool(stage.get("used_llm"))
        badge_cls = "stage-badge--live" if used_llm else "stage-badge--heuristic"
        badge_text = "AI" if used_llm else "Smart rules"
        note = escape(safe_text(stage.get("note")))
        body = _trace_output_summary(name, stage.get("output") or {})
        body_html = f'<div class="trace-body">{body}</div>' if body else ""
        stages.append(
            f'<div class="trace-stage">'
            f'<div class="trace-stage-head">'
            f'<span class="trace-step-no">{i}</span>'
            f'<span class="trace-stage-name">{escape(name)}</span>'
            f'<span class="stage-badge {badge_cls}">'
            f'<span class="status-dot"></span>{badge_text}</span>'
            f'</div>'
            f'<div class="trace-note">{note}</div>'
            f'{body_html}'
            f'</div>'
        )
    count = len(trace)
    return (
        f'<details class="reasoning-trace">'
        f'<summary>How we chose these · {count} steps</summary>'
        f'<div class="trace-intro">A quick look at how BookRec went from your '
        f'message to this shortlist.</div>'
        f'<div class="trace-stages">' + "".join(stages) + '</div>'
        f'</details>'
    )


def render_clarify_message(message) -> str:
    """Build the HTML for an assistant turn that asks a clarifying question.

    Shows the question as the lead plus the intent chips for what we already
    understood. The tappable options are rendered as real Streamlit buttons in
    the composer area, not here.
    """
    intent = message.get("intent") or {}
    question = escape(safe_text(message.get("question_text"), "Could you tell me a bit more?"))
    source = escape(safe_text(message.get("source"), ""))

    parts = [
        '<div class="msg msg-assistant">',
        '<div class="assistant-avatar">B</div>',
        '<div class="assistant-body">',
        '<div class="assistant-name">BookRec</div>',
        f'<div class="assistant-lead">{question}</div>',
        render_intent_chips(intent),
        '<div class="clarify-hint">Choose an option below, type your own answer, '
        'or skip straight to recommendations.</div>',
        render_reasoning_trace(message.get("trace")),
    ]
    if source:
        parts.append(f'<div class="assistant-meta">Source: {source}</div>')
    parts.append("</div>")  # assistant-body
    parts.append("</div>")  # msg
    return "\n".join(parts)


def _goodreads_url(cover_url, title, authors):
    """Best-effort link to the book's Goodreads page.

    The Goodreads cover filename is the goodreads_book_id (e.g.
    ``.../books/1447303603s/2767052.jpg`` -> 2767052), which gives a direct book
    page. When the cover is a generic placeholder (no numeric id), fall back to a
    Goodreads search by title + author so the link always works.
    """
    match = re.search(r"/(\d+)\.[a-zA-Z]+$", cover_url or "")
    if match:
        return f"https://www.goodreads.com/book/show/{match.group(1)}"
    query = urllib.parse.quote_plus(f"{title} {authors}".strip())
    return f"https://www.goodreads.com/search?q={query}"


@st.cache_data
def book_media_lookup(books):
    """Map book_id -> {"cover": <url or "">, "url": <goodreads link>}.

    Lets the chat pick cards show the same cover thumbnail (from the catalog's
    image link) and link out to Goodreads, working from just the pick's book_id.
    """
    cover_col = next(
        (c for c in ("small_image_url", "image_url") if c in books.columns), None
    )
    out = {}
    for row in books.itertuples(index=False):
        cover = safe_text(getattr(row, cover_col, "")) if cover_col else ""
        out[getattr(row, "book_id")] = {
            "cover": cover,
            "url": _goodreads_url(
                cover,
                safe_text(getattr(row, "title", "")),
                safe_text(getattr(row, "authors", "")),
            ),
        }
    return out


def render_assistant_message(message, book_meta=None) -> str:
    """Build the HTML for one assistant turn (a re-ranked shortlist)."""
    if message.get("kind") == "clarify":
        return render_clarify_message(message)
    book_meta = book_meta or {}
    picks = message.get("picks", [])
    intent = message.get("intent") or {}
    lead_text = safe_text(intent.get("summary"))
    if not lead_text:
        pref = safe_text(message.get("pref"), "your request")
        verb = "Refined the shortlist" if message.get("refine") else "Here is your shortlist"
        lead_text = f"{verb} for “{pref}”."
    lead = escape(lead_text)
    source = escape(safe_text(message.get("source"), ""))

    parts = [
        '<div class="msg msg-assistant">',
        '<div class="assistant-avatar">B</div>',
        '<div class="assistant-body">',
        '<div class="assistant-name">BookRec</div>',
        f'<div class="assistant-lead">{lead}</div>',
        render_intent_chips(intent),
        '<div class="pick-list">',
    ]
    if picks:
        for rank, pick in enumerate(picks, start=1):
            title = escape(safe_text(pick.title, "Untitled"))
            authors = escape(safe_text(pick.authors, "Unknown author"))
            why = escape(safe_text(pick.explanation, "Ranked for this preference."))
            # description is the NEW contract field — one neutral sentence about
            # the book itself. May be empty for older/edge picks, so only render
            # the line when it is populated.
            desc = safe_text(getattr(pick, "description", ""))
            desc_html = (
                f'<div class="pick-desc">{escape(desc)}</div>' if desc else ""
            )
            # Cover thumbnail + Goodreads link, looked up by the pick's book_id.
            media = book_meta.get(getattr(pick, "book_id", None), {})
            cover = safe_text(media.get("cover", ""))
            link = safe_text(media.get("url", ""))
            if cover.startswith("http"):
                cover_inner = (
                    f'<img class="pick-cover-img" src="{escape(cover)}" alt="" '
                    f'loading="lazy" referrerpolicy="no-referrer">'
                )
            else:
                cover_inner = (
                    '<span class="pick-cover-img pick-cover--empty" '
                    'aria-hidden="true">📖</span>'
                )
            if link.startswith("http"):
                cover_html = (
                    f'<a class="pick-cover" href="{escape(link)}" target="_blank" '
                    f'rel="noopener noreferrer">{cover_inner}</a>'
                )
                title_html = (
                    f'<a class="pick-title-link" href="{escape(link)}" '
                    f'target="_blank" rel="noopener noreferrer">{title}</a>'
                )
            else:
                cover_html = f'<div class="pick-cover">{cover_inner}</div>'
                title_html = title
            parts.append(
                f'<article class="pick-card">'
                f'<div class="pick-rank">{rank}</div>'
                f'{cover_html}'
                f'<div class="pick-title">{title_html}'
                f'<span class="pick-author"> by {authors}</span></div>'
                f'{desc_html}'
                f'<div class="pick-why">'
                f'<span class="pick-why-label">Why #{rank}</span>{why}'
                f'</div>'
                f'</article>'
            )
    else:
        parts.append(
            '<div class="pick-empty">No candidates matched closely enough. '
            'Try regenerating candidates or loosening the filters above.</div>'
        )
    parts.append("</div>")  # pick-list
    parts.append(render_reasoning_trace(message.get("trace")))
    if source:
        parts.append(f'<div class="assistant-meta">Source: {source}</div>')
    parts.append("</div>")  # assistant-body
    parts.append("</div>")  # msg
    return "\n".join(parts)


def render_chat_thread(messages, pending: bool = False, book_meta=None) -> None:
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
            parts.append(render_assistant_message(message, book_meta))
    if pending:
        stage_pills = "".join(
            f'<span class="stage-pill"><span class="stage-pill-dot"></span>{label}</span>'
            for label in ("Reading your request", "Rating the books", "Choosing the best")
        )
        parts.append(
            '<div class="msg msg-assistant">'
            '<div class="assistant-avatar">B</div>'
            '<div class="assistant-body">'
            '<div class="assistant-name">BookRec</div>'
            '<div class="assistant-lead">Working through your request…</div>'
            f'<div class="stage-progress">{stage_pills}</div>'
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


def auto_reader(ratings):
    """Pick a sensible default reader for UBCF: the most active rater.

    The reader is no longer a primary control — UBCF still needs a user to
    personalize for, so we default to the reader with the richest history (the
    most signal for the model). The chat does the real personalization on top.
    """
    return ratings["user_id"].value_counts().idxmax()


def _split_values(series):
    """Yield individual comma-split, stripped values from a string column."""
    for cell in series.dropna():
        for part in str(cell).split(","):
            value = part.strip()
            if value:
                yield value


@st.cache_data
def author_options(books):
    """Sorted unique individual authors (comma-split), dropping blanks/Unknown."""
    authors = {
        a for a in _split_values(books.get("authors", pd.Series(dtype=str)))
        if a and a != "Unknown"
    }
    return sorted(authors)


# A decade needs at least this many books to stand alone as a filter option;
# the run of sparser early decades is collapsed into one "Before <cutoff>s" bin
# so the UBCF candidate pool for any selected period is never trivially small.
MIN_DECADE_BOOKS = 50


@st.cache_data
def decade_options(books):
    """Decade labels for the filter — individual recent decades, sparse old ones binned.

    Decades with enough books (>= MIN_DECADE_BOOKS) are listed individually
    (e.g. "1990s"). The leading run of sparse early decades is collapsed into a
    single "Before <cutoff>s" option (cutoff = the first dense decade), so any
    selected period always has enough candidates for the collaborative filter.
    Falls back to listing every decade when there is no meaningful split (e.g.
    the small synthetic sample).
    """
    years = pd.to_numeric(
        books.get("original_publication_year", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    years = years[years > 0]
    if years.empty:
        return []
    counts = (years // 10 * 10).astype(int).value_counts()
    decades = sorted(counts.index)
    dense = [d for d in decades if counts[d] >= MIN_DECADE_BOOKS]
    if len(dense) < 2 or dense[0] == decades[0]:
        # Nothing sparse to collapse — list every decade individually.
        return [f"{d}s" for d in decades]
    cutoff = dense[0]
    return [f"Before {cutoff}s"] + [f"{d}s" for d in decades if d >= cutoff]


@st.cache_data
def genre_options(books):
    """Sorted unique genres (comma-split) if a non-empty `genre` column exists, else []."""
    if "genre" not in books.columns:
        return []
    genres = {g for g in _split_values(books["genre"]) if g}
    return sorted(genres)


def filter_book_ids(books, authors=None, decades=None, genres=None):
    """Restrict the catalog by author/decade/genre.

    Returns None when nothing is selected (a true no-op for the caller). When any
    group is active, builds a vectorized boolean mask: AND across active groups,
    OR within a group (a book matches a group if ANY of its split values is
    selected). Returns set(book_id) of the surviving rows.
    """
    authors = authors or []
    decades = decades or []
    genres = genres or []
    if not authors and not decades and not genres:
        return None

    mask = pd.Series(True, index=books.index)

    if authors:
        wanted = set(authors)
        author_split = books.get("authors", pd.Series("", index=books.index)).fillna("")
        author_mask = author_split.apply(
            lambda cell: any(
                part.strip() in wanted for part in str(cell).split(",")
            )
        )
        mask &= author_mask

    if decades:
        # Decade labels are either an exact decade ("1990s") or the collapsed
        # "Before <cutoff>s" bin; handle both (and any combination of them).
        exact_decades = set()
        before_cutoff = None
        for d in decades:
            label = str(d)
            if label.startswith("Before "):
                try:
                    before_cutoff = int(label.replace("Before ", "").rstrip("s"))
                except ValueError:
                    pass
            else:
                try:
                    exact_decades.add(int(label.rstrip("s")))
                except ValueError:
                    pass
        years = pd.to_numeric(
            books.get("original_publication_year", pd.Series(0, index=books.index)),
            errors="coerce",
        ).fillna(0)
        decade_mask = (years // 10 * 10).astype(int).isin(exact_decades) & (years > 0)
        if before_cutoff is not None:
            decade_mask = decade_mask | ((years > 0) & (years < before_cutoff))
        mask &= decade_mask

    if genres and "genre" in books.columns:
        wanted_genres = set(genres)
        genre_mask = books["genre"].fillna("").apply(
            lambda cell: any(
                part.strip() in wanted_genres for part in str(cell).split(",")
            )
        )
        mask &= genre_mask

    return set(books.loc[mask, "book_id"])


@st.cache_data
def dataset_insights(ratings, books):
    """Aggregates for the EDA / insights section (cached per dataset).

    Surfaces the patterns that actually shape modeling on this data: the
    positivity bias in ratings, the popularity long tail, the active-vs-casual
    reader skew, sparsity, and the modern-skewed catalog.
    """
    out = {}
    out["rating_dist"] = (
        ratings["rating"].round(1).value_counts().sort_index().rename("ratings")
    )
    out["mean_rating"] = float(ratings["rating"].mean())
    out["pct_4plus"] = float((ratings["rating"] >= 4).mean())

    per_user = ratings.groupby("user_id").size()
    out["per_user_median"] = int(per_user.median())
    out["per_user_mean"] = float(per_user.mean())
    out["per_user_max"] = int(per_user.max())

    per_book = ratings.groupby("book_id").size().sort_values(ascending=False)
    out["per_book_median"] = int(per_book.median())
    top10pct_n = max(1, int(len(per_book) * 0.10))
    out["top10pct_share"] = float(per_book.head(top10pct_n).sum() / per_book.sum())

    top = per_book.head(10).rename("ratings").reset_index().merge(
        books[["book_id", "title", "authors", "average_rating"]],
        on="book_id", how="left",
    )
    out["top_books"] = top[["title", "authors", "ratings", "average_rating"]]

    years = pd.to_numeric(books["original_publication_year"], errors="coerce")
    years = years[years > 0]
    decade_counts = (years // 10 * 10).astype(int).value_counts().sort_index()
    decade_counts.index = [f"{d}s" for d in decade_counts.index]
    out["decade_dist"] = decade_counts.rename("books")

    out["sparsity"] = 1 - len(ratings) / (
        ratings["user_id"].nunique() * ratings["book_id"].nunique()
    )
    return out


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


def submit_chat_message(prompt: str, top_k: int, skip_clarify: bool = False) -> None:
    """Append the user's turn and queue an assistant reply for the next run.

    ``skip_clarify`` powers the "Just recommend something" escape hatch — it tells
    the DAG to bypass the clarifying-question gate and rank immediately.
    """
    messages = st.session_state.setdefault("chat_messages", [])
    messages.append({"role": "user", "content": prompt})
    st.session_state["chat_pending"] = {"top_k": top_k, "skip_clarify": skip_clarify}
    st.session_state["clear_chat_input"] = True


def resolve_pending_chat(books) -> None:
    """Run the conversation-aware RAG re-rank for the queued user turn.

    The DAG may pause at its clarify gate and return a question instead of picks;
    we append either a ``kind:"clarify"`` or a ``kind:"recommend"`` assistant turn.
    """
    pending = st.session_state.get("chat_pending")
    if not pending:
        return
    st.session_state["chat_pending"] = None  # clear early so we never re-enter

    messages = st.session_state.get("chat_messages", [])
    recs = st.session_state.get("cf_recs")
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    if recs is None or not user_turns:
        return

    # How many clarifying questions we've already asked this conversation; the
    # gate stops asking once it hits the pipeline's max-rounds cap.
    rounds_asked = sum(1 for m in messages if m.get("kind") == "clarify")

    cands = llm_rerank.candidates_from_recs(recs, books)
    result = rag_pipeline.run_pipeline(
        user_turns,
        cands,
        top_k=pending["top_k"],
        rounds_asked=rounds_asked,
        skip_clarify=pending.get("skip_clarify", False),
    )

    if result.question is not None:
        messages.append(
            {
                "role": "assistant",
                "kind": "clarify",
                "question_text": result.question.prompt,
                "options": list(result.question.options),
                "source": result.source,
                "used_llm": result.used_llm,
                "intent": asdict(result.intent),
                "trace": [asdict(stage) for stage in result.trace],
            }
        )
        return

    messages.append(
        {
            "role": "assistant",
            "kind": "recommend",
            "picks": result.picks,
            "source": result.source,
            "used_llm": result.used_llm,
            "refine": len(user_turns) > 1,
            "pref": user_turns[-1],
            "intent": asdict(result.intent),
            "trace": [asdict(stage) for stage in result.trace],
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
        '<div class="section-copy">Filter by author and decade to shape the candidate pool the chat assistant will re-rank. Data source, model, and tuning live under Advanced options.</div>',
        unsafe_allow_html=True,
    )

    # The base view shows only the author + decade filters. They depend on the
    # loaded catalog (which depends on the data source, an advanced setting), so
    # we reserve their row here and fill it after the data has loaded below.
    base_filters = st.container()

    # Everything that used to be a primary control is now collapsed by default.
    with st.expander("Advanced options", expanded=False):
        source = st.selectbox("Data source", [SOURCE_REAL, SOURCE_SAMPLE])
        model_choices = (
            ["ubcf", "ibcf", "baseline", "svd", "popularity"]
            if HAVE_SURPRISE
            else ["popularity"]
        )
        cf_kind = st.selectbox("Model", model_choices, format_func=model_label)
        k_neighbors = st.slider("Neighborhood size", 5, 50, DEFAULT_UBCF_K)
        top_n = st.slider("Candidate count", 5, 30, 10)
        min_ratings = st.slider("Minimum ratings per book", 0, 200, DEFAULT_MIN_RATINGS)
        # The reader override needs the loaded ratings, so reserve its slot and
        # fill it once the catalog is available.
        reader_slot = st.container()

    with st.spinner("Loading catalog..."):
        ratings, books = load_data(source)

    # UBCF needs a user; default to the most active reader unless overridden.
    auto_uid = auto_reader(ratings)
    with reader_slot:
        reader_choice = st.selectbox(
            "Reader",
            ["Auto"] + sorted(ratings["user_id"].unique())[:1000],
            help="Auto uses the most active reader as the UBCF base. The chat does "
                 "the real personalization on top of these candidates.",
            key="reader_override",
        )
    uid = auto_uid if reader_choice == "Auto" else reader_choice

    author_opts = author_options(books)
    decade_opts = decade_options(books)
    genre_opts = genre_options(books)

    with base_filters:
        author_col, decade_col = st.columns(2)
        with author_col:
            sel_authors = st.multiselect("Authors", author_opts, key="filt_authors")
        with decade_col:
            sel_decades = st.multiselect("Decades", decade_opts, key="filt_decades")
        # Genre stays hidden until genre data exists (genre_opts is empty today).
        sel_genres = []
        if genre_opts:
            sel_genres = st.multiselect("Genres", genre_opts, key="filt_genres")

    allowed_book_ids = filter_book_ids(books, sel_authors, sel_decades, sel_genres)

    current_config = (
        source,
        cf_kind,
        k_neighbors,
        top_n,
        min_ratings,
        uid,
        tuple(sorted(sel_authors)),
        tuple(sorted(sel_decades)),
        tuple(sorted(sel_genres)),
    )
    if st.session_state.get("rec_config") != current_config:
        st.session_state["cf_recs"] = None
        st.session_state["chat_messages"] = []
        st.session_state["chat_pending"] = None

    filter_pills = ""
    if sel_authors:
        filter_pills += f'<span class="pill">{len(sel_authors)} author(s)</span>'
    if sel_decades:
        filter_pills += (
            f'<span class="pill">{escape(", ".join(sel_decades))}</span>'
        )
    if sel_genres:
        filter_pills += f'<span class="pill">{len(sel_genres)} genre(s)</span>'
    st.markdown(
        f'<div class="pill-row">'
        f'<span class="pill">{escape(model_label(cf_kind))}</span>'
        f'<span class="pill">Top {top_n}</span>'
        f'<span class="pill">Min {min_ratings:,} book ratings</span>'
        f'{filter_pills}'
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
                allowed_book_ids=allowed_book_ids,
            )
            st.session_state["rec_config"] = current_config
            st.session_state["chat_messages"] = []
            st.session_state["chat_pending"] = None
        if allowed_book_ids is not None and st.session_state["cf_recs"].empty:
            st.warning(
                "Your filters removed every candidate — relax a filter or lower "
                "the minimum ratings."
            )

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

st.markdown('<div id="insights" class="section"></div>', unsafe_allow_html=True)
with st.expander("Dataset insights — users, books & ratings (EDA)", expanded=False):
    ins = dataset_insights(ratings, books)
    chart_l, chart_r = st.columns(2)
    with chart_l:
        st.caption("How users rate — rating-value distribution")
        st.bar_chart(ins["rating_dist"], color="#a35421")
    with chart_r:
        st.caption("Catalog by publication decade")
        st.bar_chart(ins["decade_dist"], color="#7c4a23")
    st.caption("Most-rated books — the head of the popularity long tail")
    st.dataframe(ins["top_books"], hide_index=True, width="stretch")
    st.markdown(
        f"""
**What the data shows**

- **Positivity bias.** The mean rating is **{ins['mean_rating']:.2f} / 5** and
  **{ins['pct_4plus']:.0%}** of all ratings are 4★ or higher — people mostly log
  books they already liked. This inflates accuracy and makes a mean/popularity
  benchmark hard to beat.
- **Popularity is a long tail.** The top 10% most-rated books capture
  **{ins['top10pct_share']:.0%}** of all ratings, while the median book has only
  **{ins['per_book_median']}** ratings — a few blockbusters dominate.
- **Active vs. casual readers.** The median reader has rated
  **{ins['per_user_median']}** books (mean {ins['per_user_mean']:.0f}, max
  {ins['per_user_max']:,}) — a classic power-user skew.
- **Sparsity.** The user × book matrix is **{ins['sparsity']:.1%}** empty — most
  users haven't rated most books. This is the core challenge for collaborative
  filtering (cold-start, thin neighborhoods).
- **Modern-skewed catalog.** Most titles are post-1990; pre-1900 decades are so
  sparse they're grouped into a single "Before 1900s" filter bin.

**Why it matters for modeling:** strong positivity + popularity concentration
means a **popularity/mean baseline is a tough benchmark**, and sparsity limits
neighborhood CF — exactly why we benchmark UBCF/IBCF against the baseline
(*Advanced · Model quality & audit*) and add an LLM layer for personalization
*beyond* popularity.
"""
    )

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
        f'{len(recs):,} books matching your filters.</div>',
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
with st.expander("Advanced · Model quality & audit", expanded=False):
    st.markdown(
        "**Offline hold-out from the project notebooks** (fixed test set). "
        "UBCF (pearson) is the best model."
    )
    st.dataframe(
        pd.DataFrame(
            NOTEBOOK_METRICS,
            columns=["model", "P@10", "R@10", "F1@10"],
        ).set_index("model"),
        width="stretch",
    )
    st.markdown(
        f"Or run a fresh 90/10 hold-out bake-off in this session (seed {RANDOM_STATE})."
    )
    if st.button("Run model audit", type="primary"):
        st.session_state["eval_ran"] = True
    if st.session_state.get("eval_ran"):
        results, n_train, n_test = run_model_bakeoff(source, k_neighbors)
        st.markdown(
            f'<div class="section-copy">Live bake-off — train: {n_train:,} ratings, '
            f'test: {n_test:,} ratings.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(results, width="stretch")

st.markdown('<div id="business" class="section"></div>', unsafe_allow_html=True)
with st.expander("Business applications & recommended approach", expanded=False):
    st.markdown(
        """
**Collaborative filtering (UBCF / IBCF).** Powers the "readers like you also
enjoyed…" experience — catalog discovery, engagement, retention, and cross-sell.
It's cheap to serve once trained and needs no content metadata, just the
behavior signal the business already collects.

**LLM re-ranking layer.** Turns a static Top-N into *natural-language,
mood-aware* personalization with a short, explainable reason per pick ("why this
book"). That drives conversational commerce, merchandising, and trust — it
captures intent ("a cozy mystery, nothing gory") that ratings alone can't
express, and it differentiates the UX.

**Challenges a business would face**

- *Collaborative filtering:* cold-start for new users/books, data **sparsity**,
  popularity bias (the long tail above), scaling similarity to millions of
  users, deciding retrain cadence, and the gap between offline metrics and real
  online lift.
- *LLM layer:* API **cost & latency** at scale, **grounding** (the model must
  re-rank only real candidates, never invent books), prompt-injection/safety,
  vendor lock-in and **API-key management**, reproducibility, and measuring
  incremental value over plain CF.

**Recommended approach for this dataset.** The EDA shows a strong popularity
baseline and high sparsity, so: use **UBCF (pearson)** — the best CF model in our
audit — to generate candidates, but keep the popularity/mean baseline as a
guardrail and cold-start fallback. Layer the **LLM re-ranker** strictly on top of
those CF candidates for personalization + explanations, cache/limit LLM calls to
control cost, and graduate from offline Precision/Recall@K to **A/B-tested online
lift** once live.

*Model used for the AI layer: Google Gemini (`gemini-2.0-flash`). The API key is
read from the environment and never committed; without it the app falls back to a
transparent heuristic re-ranker so it always runs.*
"""
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
    # The DAG paused to ask a question if the last turn is a clarify message and
    # we're not mid-run; the composer then shows the question's quick-reply chips.
    last_message = messages[-1] if messages else None
    awaiting_clarify = (
        not pending
        and last_message is not None
        and last_message.get("role") == "assistant"
        and last_message.get("kind") == "clarify"
    )

    if refining:
        render_chat_thread(
            messages, pending=pending, book_meta=book_media_lookup(books)
        )
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
            [0.64, 0.18, 0.18],
            gap="small",
            vertical_alignment="center",
        )
        with control_cols[1]:
            chat_mode = st.selectbox(
                "Depth",
                ["Focused", "Extended"],
                label_visibility="collapsed",
                help="Focused returns 5 picks; Extended returns 8.",
            )
        with control_cols[2]:
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

    if awaiting_clarify:
        clarify_options = [
            o for o in (last_message.get("options") or []) if str(o).strip()
        ]
        with st.container(key="chat_suggestions"):
            if clarify_options:
                chip_cols = st.columns(len(clarify_options), gap="small")
                for col, option in zip(chip_cols, clarify_options):
                    with col:
                        if st.button(
                            option, disabled=composer_disabled, width="stretch"
                        ):
                            submit_chat_message(option, personalized_k)
                            st.rerun()
        skip_cols = st.columns([0.5, 0.22, 0.28])
        with skip_cols[2]:
            if st.button(
                "Just recommend something →",
                disabled=composer_disabled,
                width="stretch",
            ):
                submit_chat_message(
                    "Just recommend something", personalized_k, skip_clarify=True
                )
                st.rerun()
        with skip_cols[0]:
            if st.button("↻ Start a new chat", width="stretch", disabled=pending):
                st.session_state["chat_messages"] = []
                st.session_state["chat_pending"] = None
                st.session_state["clear_chat_input"] = True
                st.rerun()
    elif refining:
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
