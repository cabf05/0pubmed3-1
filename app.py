import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import re
from collections import Counter

st.set_page_config(page_title="PubMed Relevance Ranker", layout="wide")
st.title("🔍 PubMed Relevance Ranker")
st.markdown("Fetch and rank recent PubMed articles based on relevance potential using custom criteria.")

# -------------------- Inputs --------------------
st.header("Step 1: Customize the Search")

default_query = '("Endocrinology" OR "Diabetes") AND 2024/10/01:2025/06/28[Date - Publication]'
query = st.text_area("PubMed Search Query", value=default_query, height=100)

default_journals = "\n".join([
    "N Engl J Med", "JAMA", "BMJ", "Lancet", "Nature", "Science", "Cell"
])
journal_input = st.text_area("High-Impact Journals (one per line)", value=default_journals, height=150)
journals = [j.strip().lower() for j in journal_input.splitlines() if j.strip()]

default_institutions = "\n".join([
    "Harvard", "Oxford", "Mayo", "NIH", "Stanford",
    "UCSF", "Yale", "Cambridge", "Karolinska Institute", "Johns Hopkins"
])
inst_input = st.text_area("Renowned Institutions (one per line)", value=default_institutions, height=150)
institutions = [i.strip().lower() for i in inst_input.splitlines() if i.strip()]

default_summary = "\n".join([
    "Harvard","Stanford","Massachusetts Institute of Technology","University of Cambridge","University of Oxford",
    "University of California, Berkeley","Princeton University","Yale University","University of Chicago","Columbia",
    "California Institute of Technology","University College London","ETH Zurich","Imperial College London","University of Toronto",
    "Tsinghua University","Peking University","National University of Singapore","University of Melbourne","University of Tokyo",
    "Kyoto University","Seoul National University","University of Hong Kong","University of British Columbia","University of Sydney",
    "University of Edinburgh","University of Manchester","Ludwig Maximilian University of Munich","University of Copenhagen",
    "University of Amsterdam","University of Zurich","McGill University","King's College London","University of Illinois Urbana-Champaign",
    "École Polytechnique Fédérale de Lausanne","University of Pennsylvania","Cornell University","Johns Hopkins","Duke University",
    "University of California, Los Angeles","University of Michigan","University of Texas at Austin","Washington University in St. Louis",
    "University of California, San Diego","University of California, Davis","University of Washington","University of Wisconsin–Madison",
    "New York University","University of North Carolina at Chapel Hill","National Taiwan University"
])
summary_input = st.text_area(
    "Institutions for Summary Analysis (one per line)",
    value=default_summary,
    height=200
)
summary_institutions = [i.strip().lower() for i in summary_input.splitlines() if i.strip()]

default_keywords = "\n".join([
    "glp-1", "semaglutide", "tirzepatide", "ai", "machine learning", "telemedicine"
])
hot_input = st.text_area("Hot Keywords (one per line)", value=default_keywords, height=100)
hot_keywords = [k.strip().lower() for k in hot_input.splitlines() if k.strip()]

max_results = st.number_input("Max number of articles to fetch", min_value=10, max_value=1000, value=250, step=10)

# -------------------- Utility Functions --------------------
def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip().lower()

INSTITUTION_KEYWORDS = [
    "univ", "university", "hospital", "clinic", "institute",
    "college", "center", "centre", "school", "department",
    "laboratory", "lab"
]

def split_affiliations(raw_aff, institution_list):
    parts = (raw_aff or "").split(";")
    filtered = []
    for part in parts:
        text = normalize_text(part)
        if len(text) < 5 or re.fullmatch(r"\d+", text):
            continue
        if any(inst in text for inst in institution_list):
            filtered.append(text)
            continue
        if any(kw in text for kw in INSTITUTION_KEYWORDS):
            filtered.append(text)
    return list(dict.fromkeys(filtered))

def match_institution(text, institution_list):
    text = normalize_text(text)
    return any(re.search(rf"\b{re.escape(inst)}\b", text) for inst in institution_list)

# -------------------- Search and Processing --------------------
if st.button("🔎 Run PubMed Search"):
    with st.spinner("Fetching articles..."):
        # ESearch
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"pubmed","retmax":str(max_results),"retmode":"json","term":query}
        )
        id_list = r.json().get("esearchresult", {}).get("idlist", [])

        # EFetch
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db":"pubmed","id":",".join(id_list),"retmode":"xml"},
            timeout=20
        )

        parsed_ok = parsed_fail = 0
        records = []

        def score_article(article, aff_parts, title_text):
            score, reasons = 0, []
            journal = article.findtext(".//Journal/Title","").lower()
            if any(j in journal for j in journals):
                score+=2; reasons.append("High-impact journal (+2)")
            pub_types = [pt.text.lower() for pt in article.findall(".//PublicationType")]
            valued = ["randomized controlled trial","systematic review","meta-analysis","guideline","practice guideline"]
            if any(pt in valued for pt in pub_types):
                score+=2; reasons.append("Valued publication type (+2)")
            if len(article.findall(".//Author"))>=5:
                score+=1; reasons.append("Multiple authors (+1)")
            if any(match_institution(aff, institutions) for aff in aff_parts):
                score+=1; reasons.append("Prestigious institution (+1)")
            if any(kw in title_text for kw in hot_keywords):
                score+=2; reasons.append("Hot keyword in title (+2)")
            if article.find(".//GrantList") is not None:
                score+=2; reasons.append("Has research funding (+2)")
            return score, "; ".join(reasons)

        def build_citation(article):
            authors = article.findall(".//Author")
            if authors:
                first = authors[0]
                last = first.findtext("LastName","")
                init = first.findtext("Initials","")
                auth = f"{last} {init}" if last else "Unknown Author"
            else:
                auth = "Unknown Author"
            year = article.findtext(".//PubDate/Year") or "n.d."
            title = article.findtext(".//ArticleTitle","").strip()
            journal = article.findtext(".//Journal/Title","")
            return f"{auth} et al. ({year}). {title}. {journal}."

        try:
            root = ET.fromstring(response.content)
            for art in root.findall(".//PubmedArticle"):
                try:
                    pmid = art.findtext(".//PMID")
                    title = art.findtext(".//ArticleTitle","") or ""
                    link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    journal = art.findtext(".//Journal/Title","")
                    date = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or "N/A"

                    raw_affs = [a.text for a in art.findall(".//AffiliationInfo/Affiliation") if a.text]
                    aff_text = "; ".join(raw_affs)
                    aff_parts = split_affiliations(aff_text, institutions)

                    abstract_elems = art.findall(".//Abstract/AbstractText")
                    abstract = "\n".join(e.text.strip() for e in abstract_elems if e.text) if abstract_elems else "N/A"

                    pub_types = [pt.text for pt in art.findall(".//PublicationType")]
                    pub_types_text = "; ".join(pub_types)
                    citation = build_citation(art)

                    score, reason = score_article(art, aff_parts, normalize_text(title))
                    records.append({
                        "Title": title,
                        "Link": link,
                        "Journal": journal,
                        "Date": date,
                        "Publication Types": pub_types_text,
                        "Affiliations": aff_text,
                        "AffParts": aff_parts,
                        "Abstract": abstract,
                        "Citation": citation,
                        "Score": score,
                        "Why": reason
                    })
                    parsed_ok += 1
                except:
                    parsed_fail += 1
        except:
            st.error("Failed to parse XML from PubMed.")

        df = pd.DataFrame(records).sort_values("Score", ascending=False)
        st.success(f"Found {len(id_list)} PMIDs. Parsed {parsed_ok}, failed {parsed_fail}.")

        if not df.empty:
            st.dataframe(df.drop(columns="AffParts")[[
                "Title","Journal","Date","Publication Types","Affiliations",
                "Score","Why","Citation","Abstract"
            ]], use_container_width=True)
            st.download_button(
                "⬇️ Download CSV",
                data=df.drop(columns="AffParts").to_csv(index=False),
                file_name="ranked_pubmed_results.csv",
                mime="text/csv"
            )

            st.header("📊 Summary Analysis")

            # 🔬 Articles per Journal
            st.subheader("🔬 Articles per Journal")
            jc = df['Journal'].value_counts()
            st.bar_chart(jc)
            st.dataframe(jc.reset_index().rename(columns={"index":"Journal","Journal":"Count"}))

            # 🏅 Renowned Institutions Summary
            st.subheader("🏅 Renowned Institutions Summary")
            ren_counter = Counter()
            for parts in df["AffParts"]:
                match = [inst for inst in institutions if any(inst in p for p in parts)]
                if match:
                    for inst in set(match):
                        ren_counter[inst] += 1
                else:
                    ren_counter["Others"] += 1
            ren_df = (
                pd.DataFrame.from_dict(ren_counter, orient="index", columns=["Count"])
                  .rename_axis("Institution")
                  .sort_values("Count", ascending=False)
            )
            st.bar_chart(ren_df)
            st.dataframe(ren_df.reset_index())

            # Selected Institutions Summary
            st.subheader("Selected Institutions Summary")
            sel_counter = Counter()
            for parts in df["AffParts"]:
                match = [inst for inst in summary_institutions if any(inst in p for p in parts)]
                if match:
                    for inst in set(match):
                        sel_counter[inst] += 1
                else:
                    sel_counter["Others"] += 1
            sel_df = (
                pd.DataFrame.from_dict(sel_counter, orient="index", columns=["Count"])
                  .rename_axis("Institution")
                  .sort_values("Count", ascending=False)
            )
            st.bar_chart(sel_df)
            st.dataframe(sel_df.reset_index())

            # 📄 Publication Types
            st.subheader("📄 Articles per Publication Type")
            pt = df["Publication Types"].str.split("; ").explode().value_counts()
            st.bar_chart(pt)
            st.dataframe(pt.reset_index().rename(columns={"index":"Publication Type",0:"Count"}))

            # 🔥 Hot Keywords in Titles
            st.subheader("🔥 Articles with Hot Keywords in Title")
            hk = Counter()
            for title in df["Title"]:
                t = normalize_text(title)
                for kw in hot_keywords:
                    if kw in t:
                        hk[kw] += 1
            hk_df = (
                pd.DataFrame.from_dict(hk, orient="index", columns=["Count"])
                  .rename_axis("Hot Keyword")
                  .sort_values("Count", ascending=False)
            )
            st.bar_chart(hk_df)
            st.dataframe(hk_df.reset_index())

            # -------------------- Hugging Face Topic Analysis --------------------
            st.header("📝 Hugging Face Topic Analysis (Abstracts)")

            st.info(
                "To use this feature, generate a Hugging Face API token with **Fine-grained → Read** or **Write** permissions. "
                "Paste it below. Do NOT share this token publicly."
            )
            HF_API_TOKEN = st.text_input(
                "Hugging Face API Token",
                type="password",
                placeholder="Paste your token here",
                help="Token must have Fine-grained permissions (Read or Write) to call the model API."
            )

            if HF_API_TOKEN:
                HF_MODEL = "facebook/bart-large-mnli"
                headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

                abstracts = [a for a in df["Abstract"] if a != "N/A" and len(a.strip()) > 20]
                candidate_labels = hot_keywords + ["diabetes","endocrinology","glucose","insulin","AI","telemedicine"]

                topic_counter = Counter()
                with st.spinner(f"Processing {len(abstracts)} abstracts via Hugging Face API..."):
                    for i, abstract in enumerate(abstracts, 1):
                        payload = {
                            "inputs": abstract,
                            "parameters": {"candidate_labels": candidate_labels},
                            "options": {"wait_for_model": True}
                        }
                        response = requests.post(
                            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                            headers=headers,
                            json=payload,
                            timeout=30
                        )
                        if response.status_code == 200:
                            result = response.json()
                            top_label = result['labels'][0]
                            topic_counter[top_label] += 1
                        else:
                            st.warning(f"Abstract {i} failed: {response.status_code}")

                topic_df = (
                    pd.DataFrame.from_dict(topic_counter, orient="index", columns=["Count"])
                      .rename_axis("Topic")
                      .sort_values("Count", ascending=False)
                )

                st.subheader("📊 Most Frequent Topics in Abstracts")
                st.bar_chart(topic_df)
                st.dataframe(topic_df.reset_index())
            else:
                st.warning("Please enter a Hugging Face API token to run topic analysis.")
