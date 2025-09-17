# app.py
import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Tentar importar KeyBERT e SentenceTransformer (opcional)
try:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    KEYBERT_AVAILABLE = True
except Exception:
    KEYBERT_AVAILABLE = False

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="Medical Hot Topics (KeyBERT)", layout="wide")
st.title("🔎 Medical Hot Topics Explorer — KeyBERT")
st.markdown(
    "Extrai palavras-chave relevantes de artigos PubMed. "
    "Tenta usar KeyBERT (com sentence-transformers). Se não estiver disponível, faz fallback para análise por n-grams."
)

# -------------------- Sidebar / Options --------------------
st.sidebar.header("Opções")
use_keybert = st.sidebar.checkbox("Usar KeyBERT (se disponível)", value=True)
embedding_model_name = st.sidebar.text_input("Modelo de embeddings (sentence-transformers)", value="all-MiniLM-L6-v2")
top_n_keywords = st.sidebar.number_input("Top N keywords (globais)", min_value=5, max_value=50, value=20, step=5)
per_article_keywords = st.sidebar.number_input("Top n keywords por artigo (apenas para top K artigos)", min_value=0, max_value=10, value=3)
per_article_topk = st.sidebar.number_input("Aplicar per-article keywords apenas aos top K artigos", min_value=0, max_value=200, value=20)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Observação: o `sentence-transformers` baixa um modelo e pode instalar `torch`. "
    "Em ambientes com build limitado (Streamlit Cloud / Render gratuito) isso pode falhar. "
    "Se o deploy quebrar, desmarque 'Usar KeyBERT' e use o fallback por n-grams."
)

# -------------------- Inputs --------------------
st.header("Step 1: Customize the Search")

default_query = '("Endocrinology" OR "Diabetes") AND 2024/10/01:2025/09/01[Date - Publication]'
query = st.text_area("PubMed Search Query", value=default_query, height=120)

max_results = st.number_input("Max number of articles to fetch", min_value=10, max_value=500, value=150, step=10)

# -------------------- Helpers --------------------
def extract_pub_date(article):
    # tenta várias tags comuns
    date = article.findtext(".//PubDate/Year")
    if date:
        return date
    date = article.findtext(".//ArticleDate/Year")
    if date:
        return date
    date = article.findtext(".//PubDate/MedlineDate")
    if date:
        return date
    # fallback: tentar juntar Year/Month/Day se existirem
    y = article.findtext(".//Journal/JournalIssue/PubDate/Year")
    m = article.findtext(".//Journal/JournalIssue/PubDate/Month")
    d = article.findtext(".//Journal/JournalIssue/PubDate/Day")
    if y:
        return " ".join([p for p in [y, m, d] if p])
    return "N/A"

def extract_abstract(article):
    # junta vários AbstractText possíveis
    texts = []
    for abs_part in article.findall(".//Abstract/AbstractText"):
        if abs_part is not None:
            # AbstractText pode ter atributo Label
            label = abs_part.attrib.get("Label")
            part_text = (abs_part.text or "")
            if label:
                texts.append(f"{label}: {part_text}")
            else:
                texts.append(part_text)
    return " ".join(texts).strip()

# -------------------- Main: Fetch + Parse --------------------
if st.button("🔎 Run Analysis"):
    with st.spinner("Buscando PMIDs no PubMed..."):
        try:
            # Step 1: ESearch
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "retmax": str(max_results),
                "retmode": "json",
                "term": query
            }
            r = requests.get(search_url, params=search_params, timeout=30)
            r.raise_for_status()
            id_list = r.json()["esearchresult"].get("idlist", [])
        except Exception as e:
            st.error(f"Erro ao consultar PubMed (esearch): {e}")
            st.stop()

    if not id_list:
        st.warning("Nenhum PMID retornado para a query informada.")
        st.stop()

    with st.spinner("Buscando detalhes dos artigos (efetch)..."):
        try:
            efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml"
            }
            response = requests.get(efetch_url, params=params, timeout=60)
            response.raise_for_status()
        except Exception as e:
            st.error(f"Erro ao baixar dados do PubMed (efetch): {e}")
            st.stop()

    # Parse XML
    records = []
    try:
        root = ET.fromstring(response.content)
        articles = root.findall(".//PubmedArticle")
        for article in articles:
            pmid = article.findtext(".//PMID") or ""
            title = article.findtext(".//ArticleTitle") or ""
            abstract = extract_abstract(article)
            journal = article.findtext(".//Journal/Title") or ""
            date = extract_pub_date(article)
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            records.append({
                "PMID": pmid,
                "Title": title,
                "Abstract": abstract,
                "Journal": journal,
                "Date": date,
                "Link": link
            })
    except Exception as e:
        st.error(f"Falha ao parsear XML: {e}")
        st.stop()

    df = pd.DataFrame(records)
    st.success(f"Buscados {len(df)} artigos. Exibindo resultados.")

    if df.empty:
        st.warning("Nenhum artigo válido para processar.")
        st.stop()

    # Mostrar tabela básica
    st.subheader("Artigos retornados")
    st.dataframe(df[["PMID", "Title", "Journal", "Date"]], use_container_width=True)

    # Preparar texto para análise
    # preferimos usar Title + Abstract (quando disponível)
    df["Text"] = (df["Title"].fillna("") + ". " + df["Abstract"].fillna("")).str.strip()
    corpus = " ".join(df["Text"].astype(str).tolist())
    if not corpus:
        st.warning("Não há texto suficiente para análise de keywords.")
        st.stop()

    # -------------------- KeyBERT path --------------------
    did_keybert_run = False
    if use_keybert and KEYBERT_AVAILABLE:
        with st.spinner("Carregando modelo de embeddings e KeyBERT (pode demorar na primeira vez)..."):
            try:
                # Cache do modelo de embeddings para evitar reloads em interações
                @st.cache_resource(show_spinner=False)
                def load_sentence_model(name):
                    return SentenceTransformer(name)

                sentence_model = load_sentence_model(embedding_model_name)
                kw_model = KeyBERT(sentence_model)
                # Extrair keywords globais (corpus)
                global_keywords = kw_model.extract_keywords(
                    corpus,
                    keyphrase_ngram_range=(1, 2),
                    stop_words="english",
                    top_n=top_n_keywords,
                    use_mmr=True,
                    diversity=0.6
                )
                did_keybert_run = True
            except Exception as e:
                st.warning(
                    "Não foi possível carregar KeyBERT / sentence-transformers neste ambiente. "
                    "Faremos fallback para análise por n-grams. "
                    f"Erro: {e}"
                )
                did_keybert_run = False

    if did_keybert_run:
        st.header("🔑 KeyBERT — Top keywords (global)")
        kw_df = pd.DataFrame(global_keywords, columns=["Keyword", "Score"])
        st.dataframe(kw_df, use_container_width=True)

        # WordCloud a partir das keywords (multiplicando pelo score)
        term_freq = {k: float(s) for k, s in global_keywords}
        wc = WordCloud(width=900, height=400, background_color="white").generate_from_frequencies(term_freq)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Per-article keywords (aplicar apenas nos top K artigos para economizar tempo)
        if per_article_keywords > 0 and per_article_topk > 0:
            st.subheader(f"Top {per_article_keywords} keywords por artigo (apenas para os top {per_article_topk} artigos)")
            # ordena artigos por presença de keywords? aqui usamos comprimento do texto como proxy e pegamos top k
            top_articles = df.head(per_article_topk).copy()
            per_article_results = []
            for idx, row in top_articles.iterrows():
                try:
                    text = row["Text"]
                    if not text.strip():
                        per_article_results.append((row["PMID"], row["Title"], []))
                        continue
                    kws = kw_model.extract_keywords(
                        text,
                        keyphrase_ngram_range=(1, 2),
                        stop_words="english",
                        top_n=per_article_keywords,
                        use_mmr=False
                    )
                    per_article_results.append((row["PMID"], row["Title"], kws))
                except Exception as e:
                    per_article_results.append((row["PMID"], row["Title"], []))

            # Mostrar em tabela
            rows = []
            for pmid, title, kws in per_article_results:
                rows.append({
                    "PMID": pmid,
                    "Title": title,
                    "Keywords": ", ".join([k for k, s in kws])
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # -------------------- Fallback: CountVectorizer (n-grams) --------------------
    if not did_keybert_run:
        st.header("📊 Análise por n-grams (fallback)")
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_features=50)
        X = vectorizer.fit_transform(df["Text"].astype(str).tolist())
        freqs = zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0))
        freq_df = pd.DataFrame(freqs, columns=["Keyword", "Frequency"]).sort_values("Frequency", ascending=False)
        st.dataframe(freq_df, use_container_width=True)

        # WordCloud
        freq_dict = dict(zip(freq_df["Keyword"], freq_df["Frequency"]))
        if freq_dict:
            wc = WordCloud(width=900, height=400, background_color="white").generate_from_frequencies(freq_dict)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    # -------------------- Download CSV --------------------
    st.markdown("---")
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Baixar CSV com artigos", data=csv, file_name="pubmed_articles.csv", mime="text/csv")

    st.success("Análise concluída.")
