#!/usr/bin/env python3
"""
Render the BookRec Project 2 deck to a PDF — the same MBB-consulting design built
in Figma (navy + single blue accent, action titles, exhibit cards, source/page
footers), embedding the EDA figures from `deliverable images/`.

Output: BookRec_Project2_Deck.pdf  (10 slides, 1920x1080, account-independent)
Run:    C:/Python314/python.exe make_deck_pdf.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "deliverable images")
W, H = 1920, 1080
FD = "C:/Windows/Fonts/"

# --- palette (RGB) ---
NAVY=(16,42,67); ACCENT=(47,107,255); ACCENT_LT=(107,158,255); INK=(26,39,51)
MUTED=(92,107,122); HAIR=(212,218,224); WHITE=(255,255,255); SURFACE=(244,247,250)
SOFT=(219,229,242); MUTED_D=(150,168,186)
MOTIF=(20,50,91); RING=(32,75,161)  # precomputed blends over navy

def F(kind, size):
    f = {"bold":"segoeuib.ttf","reg":"segoeui.ttf","light":"segoeuil.ttf",
         "sl":"segoeuisl.ttf"}[kind]
    return ImageFont.truetype(FD+f, size)

def tracked(d, x, y, text, font, fill, tr=2):
    """Draw letter-spaced text (for MBB eyebrows)."""
    cx = x
    for ch in text:
        d.text((cx, y), ch, font=font, fill=fill)
        cx += d.textlength(ch, font=font) + tr
    return cx

def wrap_lines(d, text, font, maxw):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        t = (cur+" "+w).strip()
        if d.textlength(t, font=font) <= maxw:
            cur = t
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

def block(d, x, y, text, font, fill, maxw, leading):
    for i, ln in enumerate(wrap_lines(d, text, font, maxw)):
        d.text((x, y+i*leading), ln, font=font, fill=fill)
    return y + len(wrap_lines(d, text, font, maxw))*leading

def chrome(d, eyebrow, title, page, src):
    d.rectangle([110,72,136,98], fill=NAVY)
    tracked(d, 150, 74, eyebrow, F("bold",22), ACCENT, 2)
    block(d, 110, 112, title, F("bold",40), NAVY, 1700, 50)
    d.rectangle([110,210,230,215], fill=ACCENT)
    d.rectangle([110,1012,1810,1013], fill=HAIR)
    d.text((110,1026), src, font=F("reg",18), fill=MUTED)
    d.text((1810,1026), "BookRec · Project 2     "+page, font=F("bold",18),
           fill=MUTED, anchor="ra")

def new_slide(bg=WHITE):
    im = Image.new("RGB", (W,H), bg)
    return im, ImageDraw.Draw(im)

slides = []

# ---------- 1. TITLE ----------
im, d = new_slide(NAVY)
d.ellipse([1290,560,1290+920,560+920], fill=MOTIF)
d.ellipse([1150,360,1150+360,360+360], outline=RING, width=2)
d.rectangle([160,300,190,330], fill=ACCENT)
tracked(d, 206, 302, "OPAN 6604   ·   PROJECT 2", F("bold",24), ACCENT_LT, 2)
d.text((154,344), "BookRec", font=F("bold",150), fill=WHITE)
d.text((164,548), "A Goodreads Book Recommender", font=F("light",52), fill=SOFT)
d.rectangle([164,636,304,642], fill=ACCENT)
block(d, 164, 676,
      "Exploratory data analysis  →  collaborative filtering vs. a popularity "
      "baseline  →  an LLM re-ranking layer that personalizes the Top-N.",
      F("reg",30), MUTED_D, 1160, 44)
d.rectangle([164,936,1304,937], fill=(70,90,115))
d.text((164,958), "Larry   ·   Erika   ·   Gloria", font=F("bold",27), fill=WHITE)
d.text((164,1000), "June 2026", font=F("reg",27), fill=MUTED_D)
slides.append(im)

# ---------- 2. EXECUTIVE SUMMARY ----------
im, d = new_slide()
chrome(d, "EXECUTIVE SUMMARY",
       "A sparse, blockbuster-driven, positively-skewed catalog makes a strong "
       "baseline hard to beat — so we pair collaborative filtering with an AI "
       "re-ranking layer.", "02",
       "Source: Project 2 ratings sample (Goodreads-derived) · 164,728 ratings · "
       "loaded via bookrec.data_loader")
def exec_card(x, y, w, h, num, stat, label, desc):
    d.rounded_rectangle([x,y,x+w,y+h], radius=12, fill=SURFACE, outline=HAIR, width=1)
    d.rounded_rectangle([x+34,y+34,x+88,y+88], radius=10, fill=ACCENT)
    d.text((x+61,y+59), num, font=F("bold",26), fill=WHITE, anchor="mm")
    d.text((x+108,y+30), stat, font=F("bold",44), fill=NAVY)
    d.text((x+110,y+90), label, font=F("bold",21), fill=ACCENT)
    block(d, x+34, y+130, desc, F("reg",21), MUTED, w-68, 29)
X,Wd,GX,Y,Hd,GY = 110,830,40,300,232,28
exec_card(X,Y,Wd,Hd,"1","98.5%","matrix sparsity","1,192 readers × 9,229 books — collaborative filtering must work from very little overlap.")
exec_card(X+Wd+GX,Y,Wd,Hd,"2","6%","top 1% of books","A blockbuster long tail (Gini 0.60). We report catalog coverage, not just accuracy.")
exec_card(X,Y+Hd+GY,Wd,Hd,"3","65%","ratings are 4–5 stars","Mean 3.84/5. RMSE rewards predicting “high,” so a popularity baseline is a serious benchmark.")
exec_card(X+Wd+GX,Y+Hd+GY,Wd,Hd,"4","735","books never rated","Plus 54% with <10 ratings. Cold-start is an item problem; user cold-start is ~0 by construction.")
by = Y+2*Hd+2*GY+10
d.rounded_rectangle([X,by,X+1700,by+134], radius=12, fill=NAVY)
d.rectangle([X,by+12,X+6,by+122], fill=ACCENT)
tracked(d, X+40, by+28, "RECOMMENDED APPROACH", F("bold",20), ACCENT_LT, 2)
block(d, X+40, by+62,
      "UBCF generates candidates · a popularity/bias baseline is the guardrail & "
      "cold-start fallback · an LLM layer adds explainable personalization — "
      "graduate from offline P/R@K to A/B-tested online lift.",
      F("bold",23), WHITE, 1620, 30)
slides.append(im)

# ---------- 3–7. EXHIBITS ----------
EXHIBITS = [
    ("exec_01_kpi_scorecard.png","EXHIBIT 1 · EXPLORATORY DATA ANALYSIS",
     "The dataset is small, dense-by-construction, and highly concentrated.","03"),
    ("exec_02_long_tail.png","EXHIBIT 2 · THE DISCOVERY PROBLEM",
     "A few blockbusters dominate — the top 1% of books capture 6% of all ratings.","04"),
    ("exec_03_rating_skew.png","EXHIBIT 3 · FEEDBACK IS NOT NEUTRAL",
     "65% of ratings are 4–5★, so a popularity baseline is genuinely hard to beat.","05"),
    ("exec_04_cold_start.png","EXHIBIT 4 · COLD-START RISK",
     "The cold-start risk sits on the books, not the readers.","06"),
    ("model_comparison.png","EXHIBIT 5 · MODEL EVALUATION",
     "UBCF wins the Top-N ranking metrics; the baseline wins RMSE — so we optimize for ranking.","07"),
]
EX_SRC = "Source: bookrec/reports · run_eda.py (figures) · held-out evaluation, notebooks (model bake-off)"
for fn, eye, title, page in EXHIBITS:
    im, d = new_slide()
    chrome(d, eye, title, page, EX_SRC)
    cx,cy,cw,ch = 291,244,1337,752
    d.rounded_rectangle([cx,cy,cx+cw,cy+ch], radius=8, fill=SURFACE, outline=HAIR, width=1)
    chart = Image.open(os.path.join(IMG, fn)).convert("RGB").resize((cw-4,ch-4), Image.LANCZOS)
    im.paste(chart, (cx+2, cy+2))
    slides.append(im)

# ---------- 8. AI RE-RANKING DAG ----------
im, d = new_slide()
chrome(d, "AI PERSONALIZATION LAYER (WEEK 4)",
       "An LLM re-ranks — never invents — the CF Top-N, grounded and explained.",
       "08", "Source: bookrec/src/rag_pipeline.py · docs/DAG_CHATBOT.md")
tracked(d, 110, 284, "FROM THE CF TOP-N (UBCF)", F("bold",19), ACCENT, 1)
stages = [("1","Understand intent","Extract mood, genre, pace & themes from the whole conversation."),
          ("2","Clarify if vague","Ask one question when the request is too thin to rank well."),
          ("3","Score candidates","Rate each CF candidate 0–1 for fit; flag avoid-terms."),
          ("4","Re-rank & explain","Order the best picks, each with a one-line reason.")]
y,hh,w = 336,232,395; xs=[110,545,980,1415]
for i,(n,nm,desc) in enumerate(stages):
    x=xs[i]
    d.rounded_rectangle([x,y,x+w,y+hh], radius=14, fill=SURFACE, outline=HAIR, width=1)
    d.rounded_rectangle([x+28,y+28,x+74,y+74], radius=10, fill=ACCENT)
    d.text((x+51,y+51), n, font=F("bold",24), fill=WHITE, anchor="mm")
    d.text((x+86,y+34), nm, font=F("bold",24), fill=NAVY)
    block(d, x+28, y+100, desc, F("reg",19), MUTED, w-56, 26)
    if i<3: d.text((x+w+8,y+86), "→", font=F("bold",40), fill=ACCENT)
tracked(d, 110, 600, "… TO A PERSONALIZED, EXPLAINED TOP-N SHOWN IN CHAT", F("bold",19), MUTED, 1)
tracked(d, 110, 668, "DESIGN PRINCIPLES", F("bold",19), ACCENT, 2)
prins=[("Structured output","Gemini response_schema → typed JSON, no parsing"),
       ("Grounded","tied to books the reader already rated highly"),
       ("No hallucination","re-ranks the candidate set; never invents books"),
       ("Resilient","heuristic fallback + in-process response caching")]
py,pw,ph=712,402,150; pxs=[110,542,974,1406]
for i,(t,s) in enumerate(prins):
    x=pxs[i]
    d.rounded_rectangle([x,py,x+pw,py+ph], radius=12, fill=WHITE, outline=HAIR, width=1)
    d.rectangle([x,py+8,x+6,py+ph-8], fill=ACCENT)
    d.text((x+28,py+26), t, font=F("bold",23), fill=NAVY)
    block(d, x+28, py+66, s, F("reg",18), MUTED, pw-50, 24)
d.rounded_rectangle([110,892,1810,978], radius=12, fill=NAVY)
tracked(d, 150, 912, "MODEL & SAFETY", F("bold",19), ACCENT_LT, 1)
block(d, 150, 944, "Google Gemini 2.5-flash-lite · API key read from the environment, "
      "never committed · falls back to a transparent heuristic when no key is set.",
      F("bold",21), WHITE, 1620, 26)
slides.append(im)

# ---------- 9. BUSINESS ----------
im, d = new_slide()
chrome(d, "BUSINESS DISCUSSION",
       "Collaborative filtering drives discovery; the LLM layer adds explainable "
       "personalization.", "09",
       "Source: bookrec/app.py (Business applications) · project write-up")
def biz_col(x, title, apps, chals):
    w,y,h = 830,290,520
    d.rounded_rectangle([x,y,x+w,y+h], radius=14, fill=SURFACE, outline=HAIR, width=1)
    d.rounded_rectangle([x,y,x+w,y+72], radius=14, fill=NAVY)
    d.rectangle([x,y+36,x+w,y+72], fill=NAVY)
    d.text((x+30,y+20), title, font=F("bold",26), fill=WHITE)
    yy=y+100
    tracked(d, x+30, yy, "APPLICATIONS", F("bold",18), ACCENT, 1); yy+=34
    for a in apps:
        d.rectangle([x+30,yy+9,x+39,yy+18], fill=ACCENT)
        ny=block(d, x+52, yy, a, F("reg",20), MUTED, w-90, 28); yy=ny+14
    yy+=6
    tracked(d, x+30, yy, "CHALLENGES", F("bold",18), ACCENT, 1); yy+=34
    for a in chals:
        d.rectangle([x+30,yy+9,x+39,yy+18], fill=ACCENT)
        ny=block(d, x+52, yy, a, F("reg",20), MUTED, w-90, 28); yy=ny+14
biz_col(110,"Collaborative filtering (UBCF / IBCF)",
        ["“Readers like you also enjoyed…” discovery, cross-sell & retention",
         "Cheap to serve once trained; needs only behavior, no metadata"],
        ["Cold-start for new users/books; data sparsity; popularity bias",
         "Offline metrics (RMSE, P/R@K) don’t guarantee online lift"])
biz_col(980,"LLM re-ranking layer",
        ["Mood- & intent-aware personalization with a reason per pick",
         "Conversational discovery and merchandising; builds user trust"],
        ["API cost & latency at scale; rate limits",
         "Grounding & safety — must re-rank real candidates, never invent"])
by=852
d.rounded_rectangle([110,by,1810,by+118], radius=12, fill=NAVY)
d.rectangle([110,by+12,116,by+106], fill=ACCENT)
tracked(d, 150, by+24, "RECOMMENDED APPROACH", F("bold",19), ACCENT_LT, 2)
block(d, 150, by+56, "UBCF candidates + a popularity/bias baseline as guardrail + "
      "the LLM re-ranker for explainable personalization; cache LLM calls to "
      "control cost; graduate from offline P/R@K to A/B-tested online lift.",
      F("bold",23), WHITE, 1620, 28)
slides.append(im)

# ---------- 10. APPENDIX DIVIDER ----------
im, d = new_slide(NAVY)
d.ellipse([1330,560,1330+900,560+900], fill=MOTIF)
d.ellipse([1180,330,1180+320,330+320], outline=RING, width=2)
d.rectangle([160,420,190,450], fill=ACCENT)
tracked(d, 206, 422, "FOR REFERENCE", F("bold",24), ACCENT_LT, 2)
d.text((154,458), "Technical Appendix", font=F("bold",96), fill=WHITE)
d.rectangle([164,612,304,618], fill=ACCENT)
block(d, 164, 652, "Nine detailed exploratory exhibits (A1–A9): matrix sparsity, "
      "concentration & Gini, Zipf fit, bias decomposition, Bayesian shrinkage, "
      "engagement, vintage, metadata completeness, and the data-quality audit.",
      F("reg",30), SOFT, 1160, 42)
slides.append(im)

# ---------- save PDF ----------
out = os.path.join(ROOT, "BookRec_Project2_Deck.pdf")
slides[0].save(out, "PDF", save_all=True, append_images=slides[1:], resolution=96.0)
print("saved", out, "—", len(slides), "slides")
