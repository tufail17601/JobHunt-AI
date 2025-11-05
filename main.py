
# AI Career Coach - Streamlit + LangChain
import os
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import initialize_agent, AgentType, Tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import SerpAPIWrapper

#  SETUP :
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Initialize model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# TOOLS :

def summarize_resume(_: str) -> str:
    """Loads and summarizes a locally saved resume file."""
    file_path = "uploaded_resume.pdf"  
    
   
    if not os.path.exists(file_path):
        return "❌ Resume file not found. Please upload your resume first."
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text = "\n\n".join([d.page_content for d in docs])

    prompt = (
        "Summarize the key skills, experiences, and strengths in this resume:\n\n" + text
    )
    summary = llm.invoke(prompt)
    return summary.content if hasattr(summary, "content") else summary


def search_jobs(query: str) -> str:
    """Fetch real job listings from Arbeitnow API."""
    url = "https://arbeitnow.com/api/job-board-api"
    response = requests.get(url)
    data = response.json().get("data", [])
    
    results = []
    for job in data:
        if query.lower() in job["title"].lower():
            results.append(f"- {job['title']} at {job['company_name']} ({job['location']})")
    return "\n".join(results[:10]) if results else "No matching jobs found."

serp = SerpAPIWrapper()
def web_search(query: str) -> str:
    """Use SerpAPI to get web search results."""
    return serp.run(query)


def linkedin_job_search(query: str) -> str:
    """Search LinkedIn or Indeed jobs using Google search via SerpAPI."""
    search_query = f"site:linkedin.com/jobs OR site:indeed.com/jobs {query}"
    results = serp.run(search_query)
    return results

def real_job_search(query: str) -> str:
    """Fetch jobs from LinkedIn, Indeed, etc. via JSearch API."""
    import requests, os

    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    params = {"query": query, "page": "1"}
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        return f"❌ API request failed: {response.text}"

    data = response.json().get("data", [])
    if not data:
        return "No jobs found for your search."

    results = []
    for job in data[:5]:
        results.append(
            f"**{job['job_title']}** at *{job['employer_name']}*  \n"
            f" {job.get('job_city', 'N/A')}, {job.get('job_country', 'N/A')}  \n"
            f"🔗 [Apply Here]({job['job_apply_link']})  \n"
        )
    
    return "\n\n".join(results)

# Register tools:
tools = [
    Tool(name="WebSearch", func=web_search, description="Search the web for information."),
    Tool(name="ResumeSummarizer", func=summarize_resume, description="Summarize resume PDF."),
    Tool(name="JobSearch", func=search_jobs, description="Search real job listings online."),
    Tool(name="LinkedInJobSearch",func=linkedin_job_search,description="Search LinkedIn or Indeed job listings using Google Search."),
    Tool(name="RealJobSearch",func=real_job_search,description="Get live job postings with links from LinkedIn/Indeed using JSearch API.")
]

# Initialize agent:
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# STREAMLIT UI :

st.set_page_config(page_title="AI Career Coach", page_icon="🧠", layout="wide")

# Sidebar:
st.sidebar.title("📂 Resume Upload")
st.sidebar.write("Upload your resume to get started:")

uploaded_file = st.sidebar.file_uploader("📄 Choose PDF", type=["pdf"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Upload a clean, text-based resume for best results.")

# Main Content:
st.title("🧠 AI Career Coach")
st.markdown(
    """
    Welcome to your personal AI-powered career coach.  
    This app analyzes your **resume**, finds **real job listings**,  
    and explains **why you're a good fit** — 
    """
)

# Input field in two-columns:
col1, col2 = st.columns([2, 1])

with col1:
    job_query = st.text_input("💼 Desired Job Title", placeholder="e.g., Machine Learning Engineer")

with col2:
    start_search = st.button("🚀 Find Jobs & Analyze Fit", use_container_width=True)

# Logic:
if start_search:
    if uploaded_file and job_query:
        with st.spinner("🔍 Analyzing your resume and searching jobs..."):
           
            temp_path = "uploaded_resume.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
                
            user_prompt = f"I have uploaded my resume. Find me {job_query} jobs and tell me why I might be a good fit."
            result = agent.run(user_prompt)

            st.success("✅ Analysis Complete!")

            st.subheader("🎯 Career Insights")
            st.write(result)

            os.remove(temp_path)
    else:
        st.warning(" Please upload your resume and enter a job title first.")

# Footer:
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Made By  ❤️Muhammad Tufail (using LangChain, Streamlit, and OpenAI)</div>",
    unsafe_allow_html=True
)
