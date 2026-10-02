import os
import streamlit as st
from rag.rag_engine import LocalRAG, load_sources

st.set_page_config(page_title="SolarGrid AI", page_icon="☀️", layout="wide")

st.markdown("""<style>
.block-container{max-width:1450px;padding-top:2rem}
.sg-card{padding:1rem 1.2rem;border:1px solid rgba(255,255,255,.12);
border-radius:16px;background:rgba(255,255,255,.045);margin:.7rem 0}
.agent{padding:.85rem 1rem;border-radius:12px;border:1px solid rgba(255,255,255,.10);margin:.45rem 0}
.small{color:#9aa4b2;font-size:.86rem}
</style>""", unsafe_allow_html=True)

st.title("☀️ SolarGrid AI")
st.caption("Pakistan Renewable Energy Intelligence & Preliminary Feasibility Platform")

with st.sidebar:
    st.header("Project")
    project=st.text_input("Project name","Multan Solar Project")
    location=st.text_input("Location","Multan, Punjab")
    technology=st.selectbox("Technology",["Solar PV"])
    capacity=st.number_input("Capacity (MW)",0.1,5000.0,50.0,1.0)
    land=st.number_input("Land area (acres)",1.0,100000.0,250.0,5.0)
    grid=st.selectbox("Grid connection",["11 kV","33 kV","66 kV","132 kV","220 kV","Other"])
    capex=st.number_input("Estimated CAPEX (PKR)",0.0,1e14,7.5e9,1e8)
    life=st.number_input("Project life (years)",1,50,25)

tabs=st.tabs(["🧭 Command Center","📚 Evidence Center","🛠️ RAG Diagnostics"])

with tabs[0]:
    a,b,c,d=st.columns(4)
    a.metric("Capacity",f"{capacity:g} MW"); b.metric("Location",location)
    c.metric("Grid",grid); d.metric("Life",f"{life} yr")
    st.subheader("AI Feasibility Workflow")
    for icon,name,desc in [
        ("✓","Project Intake","Project information validated"),
        ("○","Technical Engineer Agent","Solar generation and engineering assumptions"),
        ("○","Grid Integration Agent","Grid requirements and interconnection evidence"),
        ("○","Financial Analyst Agent","CAPEX, revenue and project economics"),
        ("○","Regulatory Intelligence Agent","Pakistan regulatory/policy retrieval"),
        ("○","Risk Analyst Agent","Technical, grid, financial and regulatory risks"),
        ("○","Project Manager Agent","Evidence-backed synthesis and report")]:
        st.markdown(f'<div class="agent"><b>{icon} {name}</b><br><span class="small">{desc}</span></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="sg-card"><b>{project}</b><br>{technology} · {location} · {capacity:g} MW · {land:g} acres · {grid}<br>Estimated CAPEX: PKR {capex:,.0f}</div>',unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Evidence Retrieval")
    query=st.text_input("Question","What are the technical requirements for connecting a generation facility to the grid?")
    k=st.slider("Retrieved passages",1,8,5)
    if st.button("🔎 Retrieve Evidence",type="primary"):
        rag=LocalRAG(); results=rag.search(query,k)
        if not results:
            st.warning("No local PDFs are indexed yet. Add the official PDFs at the paths listed in data/sources.json.")
        for r in results:
            st.markdown(f'<div class="sg-card"><b>{r["title"]}</b><br><span class="small">{r["authority"]} · {r["date"]} · relevance {r["score"]}</span><hr>{r["text"]}</div>',unsafe_allow_html=True)
            st.caption(f"Official source: {r['url']}")

with tabs[2]:
    st.subheader("RAG Diagnostics")
    rag=LocalRAG(); s=rag.status()
    c1,c2=st.columns(2); c1.metric("Indexed passages",s["indexed_passages"]); c2.metric("Indexed sources",s["indexed_sources"])
    for src in load_sources():
        path=os.path.join(os.path.dirname(__file__),src["local_file"])
        st.write(("🟢" if os.path.isfile(path) else "⚪"),src["title"],"—",src["authority"],"—",src["status"])
    st.info("The source registry is deliberately broader than the deployed local corpus. Very large or hard-to-download official documents will be handled separately rather than bloating the Streamlit deployment.")
