# ================================================================
# ADVANCED CLINICAL LITERATURE SYNTHESIS AGENT
# Enhanced with: Structured Analysis, Quality Assessment, Multi-criteria Filtering
# UPDATED: Added working filter integration (year range, study types, quality threshold)
# ================================================================

import os
import warnings
from typing import TypedDict, List, Dict
from Bio import Entrez
import streamlit as st
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import time
from datetime import datetime
import re

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION
# ================================================================

class Config:
    ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "kashishmudliar@gmail.com")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))
    WORKING_MODEL = 'models/gemini-2.5-flash-preview-05-20'
    
    GENERATION_CONFIG = {
        'temperature': 0.2,  # Lower for more factual output
        'top_p': 0.8,
        'top_k': 40,
        'max_output_tokens': 8192,
    }

# ================================================================
# STATE - UPDATED WITH FILTERS
# ================================================================

class AgentState(TypedDict):
    user_query: str
    search_results: List[Dict[str, str]]
    parsed_documents: List[Dict[str, str]]
    quality_scores: List[Dict[str, any]]
    final_summary: str
    comparative_table: str
    error: str
    metadata: Dict
    filters: Dict  # NEW: Store user filter preferences

# ================================================================
# ENHANCED NODES
# ================================================================

def search_node(state: AgentState) -> AgentState:
    """
    Enhanced PubMed search with user-defined filters
    
    NEW FEATURES:
    - Filters by publication year range
    - Filters by study type (RCT, Meta-Analysis, etc.)
    - Uses PubMed's advanced query syntax
    """
    try:
        Entrez.email = Config.ENTREZ_EMAIL
        
        # Get filters from state (NEW)
        filters = state.get('filters', {})
        year_range = filters.get('year_range', (2020, 2025))
        preferred_types = filters.get('study_types', [])
        
        search_term = state['user_query']
        
        # Build filtered query (NEW)
        query_parts = [search_term]
        
        # Add publication type filters if specified
        if preferred_types:
            type_mapping = {
                "RCT": "Randomized Controlled Trial[pt]",
                "Meta-Analysis": "Meta-Analysis[pt]",
                "Cohort": "Cohort Studies[MeSH]",
                "Case-Control": "Case-Control Studies[MeSH]"
            }
            type_filters = [type_mapping[t] for t in preferred_types if t in type_mapping]
            if type_filters:
                query_parts.append(f"({' OR '.join(type_filters)})")
        
        final_query = ' AND '.join(query_parts)
        
        # Search with filters applied
        handle = Entrez.esearch(
            db="pubmed",
            term=final_query,
            retmax=Config.MAX_RESULTS,
            sort='relevance',
            mindate=str(year_range[0]),  # NEW: Year filter
            maxdate=str(year_range[1])   # NEW: Year filter
        )
        record = Entrez.read(handle)
        handle.close()
        
        pubmed_ids = record['IdList']
        
        if not pubmed_ids:
            state['error'] = "No results found with current filters"
            return state
        
        # Fetch detailed records
        search_results = []
        handle = Entrez.efetch(db="pubmed", id=','.join(pubmed_ids), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        for record in records['PubmedArticle']:
            try:
                article = record['MedlineCitation']['Article']
                pmid = str(record['MedlineCitation']['PMID'])
                
                # Extract comprehensive metadata
                title = str(article.get('ArticleTitle', 'No title'))
                
                # Abstract with sections
                abstract_parts = article.get('Abstract', {}).get('AbstractText', [])
                if abstract_parts:
                    abstract = ' '.join([str(part) for part in abstract_parts])
                else:
                    abstract = "No abstract available"
                
                # Authors (all, not just first 3)
                authors = []
                author_list = article.get('AuthorList', [])
                for author in author_list[:5]:
                    last_name = author.get('LastName', '')
                    initials = author.get('Initials', '')
                    if last_name:
                        authors.append(f"{last_name} {initials}")
                if len(author_list) > 5:
                    authors.append("et al.")
                authors_str = ', '.join(authors) if authors else "Unknown"
                
                # Publication details
                journal_info = article.get('Journal', {})
                journal = journal_info.get('Title', 'Unknown Journal')
                pub_date = journal_info.get('JournalIssue', {}).get('PubDate', {})
                year = pub_date.get('Year', 'Unknown')
                
                # Publication type (to assess study design)
                pub_types = []
                for pub_type in article.get('PublicationTypeList', []):
                    pub_types.append(str(pub_type))
                
                # Extract study size from abstract if mentioned
                sample_size = extract_sample_size(abstract)
                
                # Get citation count from PubMed Central
                citation_count = 0
                try:
                    # Try to get citation count from PubMed link data
                    link_handle = Entrez.elink(dbfrom="pubmed", id=pmid, linkname="pubmed_pubmed_citedin")
                    link_record = Entrez.read(link_handle)
                    link_handle.close()
                    
                    if link_record and link_record[0]['LinkSetDb']:
                        citation_count = len(link_record[0]['LinkSetDb'][0]['Link'])
                except:
                    citation_count = 0
                
                search_results.append({
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract[:4000],
                    'authors': authors_str,
                    'year': str(year),
                    'journal': journal,
                    'publication_types': pub_types,
                    'sample_size': sample_size,
                    'citation_count': citation_count,
                    'link': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
                
            except Exception as e:
                continue
        
        state['search_results'] = search_results
        state['metadata'] = {
            'total_found': len(search_results),
            'search_date': datetime.now().isoformat(),
            'query': state['user_query'],
            'filters_applied': filters  # NEW: Track what filters were used
        }
        
    except Exception as e:
        state['error'] = f"Search error: {str(e)}"
    
    return state


def extract_sample_size(text: str) -> str:
    """Extract sample size from abstract using regex"""
    patterns = [
        r'(\d+)\s+patients',
        r'(\d+)\s+participants',
        r'(\d+)\s+subjects',
        r'n\s*=\s*(\d+)',
        r'N\s*=\s*(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return "Not specified"


def quality_assessment_node(state: AgentState) -> AgentState:
    """Assess evidence quality for each study"""
    try:
        quality_scores = []
        
        for doc in state['search_results']:
            score = assess_study_quality(doc)
            quality_scores.append({
                'pmid': doc['pmid'],
                'title': doc['title'],
                'quality_score': score['total'],
                'design_score': score['design'],
                'sample_score': score['sample'],
                'recency_score': score['recency'],
                'citation_score': score['citations'],
                'impact_score': score['impact'],
                'grade': score['grade'],
                'max_score': score['max_score']
            })
        
        # Sort by quality
        quality_scores.sort(key=lambda x: x['quality_score'], reverse=True)
        state['quality_scores'] = quality_scores
        
    except Exception as e:
        state['error'] = f"Quality assessment error: {str(e)}"
    
    return state


def assess_study_quality(doc: Dict) -> Dict:
    """Calculate quality score based on multiple criteria including citations and impact"""
    scores = {'design': 0, 'sample': 0, 'recency': 0, 'citations': 0, 'impact': 0}
    
    # Study design score (0-5)
    pub_types = ' '.join(doc.get('publication_types', [])).lower()
    if 'randomized controlled trial' in pub_types or 'clinical trial' in pub_types:
        scores['design'] = 5
    elif 'meta-analysis' in pub_types or 'systematic review' in pub_types:
        scores['design'] = 5
    elif 'cohort' in pub_types:
        scores['design'] = 3
    elif 'case-control' in pub_types:
        scores['design'] = 2
    else:
        scores['design'] = 1
    
    # Sample size score (0-5)
    try:
        n = int(doc.get('sample_size', '0'))
        if n >= 1000:
            scores['sample'] = 5
        elif n >= 500:
            scores['sample'] = 4
        elif n >= 100:
            scores['sample'] = 3
        elif n >= 50:
            scores['sample'] = 2
        elif n > 0:
            scores['sample'] = 1
    except:
        scores['sample'] = 0
    
    # Recency score (0-5)
    try:
        year = int(doc.get('year', '0'))
        current_year = datetime.now().year
        age = current_year - year
        if age <= 1:
            scores['recency'] = 5
        elif age <= 2:
            scores['recency'] = 4
        elif age <= 3:
            scores['recency'] = 3
        elif age <= 5:
            scores['recency'] = 2
        else:
            scores['recency'] = 1
    except:
        scores['recency'] = 0
    
    # Citation count score (0-5) - normalized by publication age
    citation_count = doc.get('citation_count', 0)
    try:
        year = int(doc.get('year', datetime.now().year))
        years_since_pub = max(1, datetime.now().year - year)
        citations_per_year = citation_count / years_since_pub
        
        if citations_per_year >= 50:
            scores['citations'] = 5
        elif citations_per_year >= 20:
            scores['citations'] = 4
        elif citations_per_year >= 10:
            scores['citations'] = 3
        elif citations_per_year >= 5:
            scores['citations'] = 2
        elif citation_count > 0:
            scores['citations'] = 1
        else:
            scores['citations'] = 0
    except:
        scores['citations'] = 0
    
    # Journal impact factor score (0-5) - estimated from journal tier
    journal = doc.get('journal', '').lower()
    impact_score = estimate_journal_impact(journal)
    scores['impact'] = impact_score
    
    total = sum(scores.values())
    
    # Enhanced GRADE classification (now 0-25 scale)
    if total >= 20:
        grade = "High Quality"
    elif total >= 14:
        grade = "Moderate Quality"
    elif total >= 8:
        grade = "Low Quality"
    else:
        grade = "Very Low Quality"
    
    scores['total'] = total
    scores['grade'] = grade
    scores['max_score'] = 25
    
    return scores


def estimate_journal_impact(journal_name: str) -> int:
    """
    Estimate journal impact tier based on journal name
    Returns score 0-5 based on journal prestige
    """
    journal_name = journal_name.lower()
    
    # Tier 1: Top-tier journals (Impact Factor > 40)
    tier1_journals = [
        'new england journal of medicine', 'nejm', 'lancet', 'jama',
        'nature', 'science', 'cell', 'nature medicine',
        'british medical journal', 'bmj'
    ]
    
    # Tier 2: High-impact specialized journals (IF 10-40)
    tier2_journals = [
        'diabetes care', 'circulation', 'journal of clinical oncology',
        'annals of internal medicine', 'plos medicine', 'jama internal medicine',
        'european heart journal', 'journal of the american college of cardiology',
        'diabetes', 'diabetologia', 'heart failure'
    ]
    
    # Tier 3: Solid specialty journals (IF 5-10)
    tier3_journals = [
        'endocrine', 'cardiovascular', 'clinical', 'medical journal',
        'american journal', 'european journal', 'international journal',
        'journal of', 'archives of'
    ]
    
    # Check tier membership
    for top_journal in tier1_journals:
        if top_journal in journal_name:
            return 5
    
    for high_journal in tier2_journals:
        if high_journal in journal_name:
            return 4
    
    for mid_journal in tier3_journals:
        if mid_journal in journal_name:
            return 3
    
    # Default tier for other peer-reviewed journals
    if journal_name and journal_name != 'unknown journal':
        return 2
    
    return 0


def filter_by_quality_node(state: AgentState) -> AgentState:
    """
    NEW NODE: Filter results based on minimum quality threshold
    
    This runs AFTER quality assessment and removes studies that don't
    meet the user's minimum quality requirement.
    """
    
    filters = state.get('filters', {})
    min_quality = filters.get('min_quality', 'Very Low')
    
    # Map quality names to minimum scores
    quality_thresholds = {
        'High': 20,
        'Moderate': 14,
        'Low': 8,
        'Very Low': 0
    }
    
    threshold = quality_thresholds.get(min_quality, 0)
    
    # Filter both search results and quality scores
    filtered_results = []
    filtered_scores = []
    
    for result in state['search_results']:
        quality = next((q for q in state['quality_scores'] if q['pmid'] == result['pmid']), None)
        
        if quality and quality['quality_score'] >= threshold:
            filtered_results.append(result)
            filtered_scores.append(quality)
    
    state['search_results'] = filtered_results
    state['quality_scores'] = filtered_scores
    
    if not filtered_results:
        state['error'] = f"No studies meet minimum quality: {min_quality}"
    
    return state


def advanced_synthesis_node(state: AgentState) -> AgentState:
    """Enhanced synthesis with structured output"""
    
    if not Config.GEMINI_API_KEY:
        state['final_summary'] = create_manual_summary(state)
        return state
    
    try:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        
        # Prepare enhanced document summaries with quality scores
        docs_text = ""
        for idx, doc in enumerate(state['search_results'][:7], 1):
            quality = next((q for q in state['quality_scores'] if q['pmid'] == doc['pmid']), {})
            
            docs_text += f"\n{'='*60}\n"
            docs_text += f"STUDY {idx}\n"
            docs_text += f"{'='*60}\n"
            docs_text += f"Title: {doc['title']}\n"
            docs_text += f"Authors: {doc['authors']}\n"
            docs_text += f"Journal: {doc['journal']} ({doc['year']})\n"
            docs_text += f"PMID: {doc['pmid']}\n"
            docs_text += f"Study Type: {', '.join(doc['publication_types'][:3])}\n"
            docs_text += f"Sample Size: {doc['sample_size']}\n"
            docs_text += f"Citations: {doc.get('citation_count', 0)}\n"
            docs_text += f"Quality Grade: {quality.get('grade', 'Not assessed')} ({quality.get('quality_score', 0)}/25)\n"
            docs_text += f"Quality Breakdown - Design: {quality.get('design_score', 0)}/5, Sample: {quality.get('sample_score', 0)}/5, Recency: {quality.get('recency_score', 0)}/5, Citations: {quality.get('citation_score', 0)}/5, Impact: {quality.get('impact_score', 0)}/5\n"
            docs_text += f"\nABSTRACT:\n{doc['abstract'][:1500]}\n"
        
        # Enhanced prompt for structured output
        prompt = f"""You are a clinical research analyst preparing an evidence synthesis report.

CLINICAL QUESTION: "{state['user_query']}"

EVIDENCE BASE: {len(state['search_results'])} studies analyzed
{docs_text}

Generate a comprehensive clinical evidence report following this EXACT structure:

# EXECUTIVE SUMMARY
[2-3 sentences directly answering the clinical question with level of evidence]

# EVIDENCE QUALITY OVERVIEW
- Total studies analyzed: {len(state['search_results'])}
- High quality studies: [count from data]
- Study designs: [RCTs, meta-analyses, cohort studies, etc.]
- Date range: [specify]

# COMPARATIVE EVIDENCE TABLE

| Study (Year) | Design | Sample Size | Citations | Impact | Key Intervention | Primary Outcome | Effect Size | Quality |
|-------------|---------|-------------|-----------|--------|------------------|-----------------|-------------|---------|
[Fill each row with ACTUAL data from the studies above - extract sample sizes, citation counts, journal impact tier, interventions, outcomes, and results]

Note: Sample Size = total number of participants in the study; Citations = total citation count from PubMed; Impact = Journal tier (5=Top tier like NEJM/Lancet, 4=High-impact specialty, 3=Solid specialty, 2=Standard peer-reviewed)

# KEY FINDINGS BY STUDY

[For EACH study provide:]
**Study [N]: [Short Title] ([Authors], [Year])**
- Design & Population: [specifics]
- Intervention: [what was tested]
- Primary Outcome: [main result with numbers]
- Limitations: [any noted in abstract]
- Quality Assessment: [reference the grade provided]

# SYNTHESIS & CLINICAL IMPLICATIONS

**Consistency of Evidence:** [Do studies agree? Any conflicts?]

**Strength of Recommendation:** [Based on quality and consistency]

**Clinical Application:** [Practical guidance for clinicians]

**Knowledge Gaps:** [What's still unclear? What future research is needed?]

# METHODOLOGICAL CONSIDERATIONS
- Study heterogeneity: [differences in populations, interventions]
- Risk of bias: [based on study designs]
- Generalizability: [who do results apply to?]

# REFERENCES
[List all studies with PMID links]

CRITICAL INSTRUCTIONS:
- Use ACTUAL numbers from the studies (sample sizes, effect sizes, p-values)
- The comparison table must have real data, not placeholders
- Be specific about outcomes (e.g., "HbA1c reduced by 0.8%", not "improved")
- Cite quality grades provided above
- If data isn't in abstract, state "not reported in abstract"
"""

        model = genai.GenerativeModel(
            model_name=Config.WORKING_MODEL,
            generation_config=Config.GENERATION_CONFIG
        )
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            state['final_summary'] = response.text
        else:
            raise Exception("Empty response")
            
    except Exception as e:
        state['error'] = f"AI synthesis error: {str(e)}"
        state['final_summary'] = create_manual_summary(state)
    
    return state


def create_manual_summary(state: AgentState) -> str:
    """Enhanced fallback summary"""
    summary = f"# Clinical Evidence Review: {state['user_query']}\n\n"
    summary += f"**{len(state['search_results'])} studies analyzed**\n\n"
    
    if state.get('error'):
        summary += f"⚠️ *{state['error']}*\n\n"
    
    # Quality overview
    if state.get('quality_scores'):
        high_quality = sum(1 for q in state['quality_scores'] if q['quality_score'] >= 12)
        summary += f"## Evidence Quality\n"
        summary += f"- High quality studies: {high_quality}\n"
        summary += f"- Total studies: {len(state['quality_scores'])}\n\n"
    
    summary += "---\n\n"
    
    for idx, doc in enumerate(state['search_results'], 1):
        quality = next((q for q in state.get('quality_scores', []) if q['pmid'] == doc['pmid']), {})
        
        summary += f"## Study {idx}: {doc['title']}\n\n"
        summary += f"- **Authors:** {doc['authors']}\n"
        summary += f"- **Journal:** {doc['journal']}\n"
        summary += f"- **Year:** {doc['year']}\n"
        summary += f"- **Sample Size:** {doc['sample_size']}\n"
        summary += f"- **Quality:** {quality.get('grade', 'Not assessed')} ({quality.get('quality_score', 0)}/15)\n"
        summary += f"- **PMID:** [{doc['pmid']}]({doc['link']})\n\n"
        summary += f"**Abstract:** {doc['abstract'][:800]}...\n\n"
        summary += "---\n\n"
    
    return summary

# ================================================================
# WORKFLOW - UPDATED WITH FILTER NODE
# ================================================================

def create_workflow() -> StateGraph:
    """
    Create the workflow with the new filter node
    
    Flow: Search → Quality Assessment → Filter by Quality → Synthesis
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("search", search_node)
    workflow.add_node("quality_assessment", quality_assessment_node)
    workflow.add_node("filter_quality", filter_by_quality_node)  # NEW NODE
    workflow.add_node("synthesis", advanced_synthesis_node)
    
    workflow.set_entry_point("search")
    workflow.add_edge("search", "quality_assessment")
    workflow.add_edge("quality_assessment", "filter_quality")  # NEW EDGE
    workflow.add_edge("filter_quality", "synthesis")           # UPDATED
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()

# ================================================================
# STREAMLIT UI - UPDATED WITH BETTER STYLING
# ================================================================

def main():
    st.set_page_config(
        page_title="Advanced Clinical Literature Synthesizer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Modern, vibrant styling
    st.markdown("""
        <style>
        /* Main app background with subtle gradient */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Sidebar with medical theme */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: white !important;
        }
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label {
            color: white !important;
        }
        
        /* Main title styling */
        h1 {
            color: #2c3e50 !important;
            font-weight: 800 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            padding: 20px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Section headers */
        h2 {
            color: #34495e !important;
            font-weight: 700 !important;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        
        h3 {
            color: #2c3e50 !important;
            font-weight: 600 !important;
        }
        
        /* Metric cards with vibrant colors */
        [data-testid="stMetricValue"] {
            color: #667eea !important;
            font-weight: 800 !important;
            font-size: 2rem !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #2c3e50 !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        [data-testid="stMetricDelta"] {
            color: #27ae60 !important;
            font-weight: 600 !important;
        }
        
        /* Text areas with modern styling */
        .stTextArea textarea {
            background-color: white !important;
            color: #2c3e50 !important;
            border: 2px solid #667eea !important;
            border-radius: 10px !important;
            font-size: 1rem !important;
            padding: 15px !important;
        }
        
        .stTextArea textarea:focus {
            border-color: #764ba2 !important;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        }
        
        .stTextArea label {
            color: #2c3e50 !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
        }
        
        /* Buttons with gradient */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 30px !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        }
        
        /* Expander headers */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%) !important;
            color: #2c3e50 !important;
            font-weight: 700 !important;
            border: 2px solid #667eea !important;
            border-radius: 10px !important;
            padding: 15px !important;
        }
        
        .streamlit-expanderHeader:hover {
            border-color: #764ba2 !important;
            background: linear-gradient(135deg, #e8eef5 0%, #dce4f0 100%) !important;
        }
        
        /* Expander content - fix text visibility */
        .streamlit-expanderContent {
            background-color: white !important;
            padding: 20px !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 0 0 10px 10px !important;
        }
        
        .streamlit-expanderContent p,
        .streamlit-expanderContent div,
        .streamlit-expanderContent span {
            color: #2c3e50 !important;
        }
        
        /* Make sure all text inside expanders is visible */
        [data-testid="stExpander"] p {
            color: #2c3e50 !important;
        }
        
        [data-testid="stExpander"] .stMarkdown {
            color: #2c3e50 !important;
        }
        
        [data-testid="stExpander"] strong {
            color: #1a1a1a !important;
            font-weight: 700 !important;
        }
        
        /* Markdown content */
        .stMarkdown {
            color: #2c3e50 !important;
        }
        
        /* Links */
        a {
            color: #667eea !important;
            font-weight: 600 !important;
            text-decoration: none !important;
        }
        
        a:hover {
            color: #764ba2 !important;
            text-decoration: underline !important;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        /* Status messages with modern cards */
        .stInfo, [data-testid="stNotification"] {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
            border-left: 5px solid #2196F3 !important;
            color: #0d47a1 !important;
            font-weight: 600 !important;
            padding: 1rem !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 6px rgba(33, 150, 243, 0.2) !important;
        }
        
        .stSuccess {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%) !important;
            border-left: 5px solid #4CAF50 !important;
            color: #1b5e20 !important;
            font-weight: 600 !important;
            padding: 1rem !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 6px rgba(76, 175, 80, 0.2) !important;
        }
        
        .stWarning {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%) !important;
            border-left: 5px solid #FF9800 !important;
            color: #e65100 !important;
            font-weight: 600 !important;
            padding: 1rem !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 6px rgba(255, 152, 0, 0.2) !important;
        }
        
        .stError {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%) !important;
            border-left: 5px solid #F44336 !important;
            color: #b71c1c !important;
            font-weight: 600 !important;
            padding: 1rem !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 6px rgba(244, 67, 54, 0.2) !important;
        }
        
        /* Progress text */
        .stProgress + div {
            color: #667eea !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
        }
        
        /* Code blocks */
        code {
            background-color: #f8f9fa !important;
            color: #e83e8c !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
        }
        
        /* Download button styling */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #27ae60 0%, #229954 100%) !important;
            color: white !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 30px !important;
            box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4) !important;
        }
        
        .stDownloadButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(39, 174, 96, 0.6) !important;
        }
        
        /* Selectbox and slider labels */
        .stSelectbox label, .stSlider label {
            color: white !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
        }
        
        /* Multiselect */
        .stMultiSelect label {
            color: white !important;
            font-weight: 700 !important;
        }
        
        /* Horizontal rule */
        hr {
            border: none;
            height: 3px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            margin: 30px 0;
        }
        
        /* Quality metric cards container */
        [data-testid="column"] {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔬 Advanced Clinical Literature Synthesis Agent")
    st.markdown("""
    **Multi-Agent Workflow:** PubMed Search → Quality Assessment → Filter by Quality → Evidence Synthesis  
    **Features:** Automated quality grading • Comparative analysis • Structured reports • Smart filtering
    """)
    
    # System status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AI Model", "Gemini 2.5 Flash", "Active")
    with col2:
        st.metric("Database", "PubMed/MEDLINE", "Connected")
    with col3:
        st.metric("Max Studies", Config.MAX_RESULTS, "Configurable")
    
    st.markdown("---")
    
    # Sidebar - UPDATED TO CAPTURE FILTER VALUES
    with st.sidebar:
        st.header("⚙️ Advanced Settings")
        
        st.markdown("**Evidence Filters:**")
        
        # Year filter (NEW - actually works now)
        year_filter = st.slider(
            "Publication Years", 
            2020, 
            2025, 
            (2020, 2025),
            help="Only show studies published in this year range"
        )
        
        # Study type filter (NEW - actually works now)
        study_types = st.multiselect(
            "Preferred Study Types",
            ["RCT", "Meta-Analysis", "Cohort", "Case-Control"],
            default=["RCT", "Meta-Analysis"],
            help="Filter PubMed search to only these study designs"
        )
        
        # Quality filter (NEW - actually works now)
        min_quality = st.select_slider(
            "Minimum Quality",
            options=["Very Low", "Low", "Moderate", "High"],
            value="Low",
            help="Hide studies below this quality threshold"
        )
        
        st.markdown("---")
        st.markdown("**System Status:**")
        if Config.GEMINI_API_KEY:
            st.success("✅ AI Analysis: Enabled")
        else:
            st.warning("⚠️ Manual Mode")
        
        st.markdown("---")
        st.markdown("### 📚 Example Queries")
        st.markdown("""
        - **Metformin efficacy type 2 diabetes**
        - **SGLT2 inhibitors heart failure mortality**
        - **Statins primary prevention cardiovascular**
        - **GLP-1 agonists weight loss diabetes**
        """)
    
    # Main query
    user_query = st.text_area(
        "📝 Enter Clinical Question:",
        height=120,
        placeholder="Example: What is the efficacy of SGLT2 inhibitors in reducing heart failure hospitalizations in patients with type 2 diabetes?",
        value="Efficacy of metformin in type 2 diabetes"
    )
    
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        analyze = st.button("🚀 Analyze Evidence", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.rerun()
    
    # Analysis - UPDATED TO PASS FILTERS TO WORKFLOW
    if analyze and user_query:
        progress = st.progress(0)
        status = st.empty()
        
        status.info("🔍 Searching PubMed database with filters...")
        progress.progress(20)
        
        # NEW: Create initial state WITH FILTERS
        initial_state = {
            "user_query": user_query,
            "search_results": [],
            "parsed_documents": [],
            "quality_scores": [],
            "final_summary": "",
            "comparative_table": "",
            "error": "",
            "metadata": {},
            "filters": {  # NEW: Pass user's filter selections
                "year_range": year_filter,
                "study_types": study_types,
                "min_quality": min_quality
            }
        }
        
        try:
            workflow = create_workflow()
            
            progress.progress(40)
            status.info("📊 Assessing evidence quality...")
            
            progress.progress(60)
            status.info("🔎 Filtering by quality threshold...")
            
            progress.progress(80)
            status.info("🤖 Synthesizing findings...")
            
            final_state = workflow.invoke(initial_state)
            
            progress.progress(100)
            status.success("✅ Analysis complete!")
            
            time.sleep(0.5)
            progress.empty()
            status.empty()
            
            # Results
            if final_state.get('error') and not final_state.get('search_results'):
                st.error(f"❌ {final_state['error']}")
            else:
                # Show applied filters (NEW)
                filters_applied = final_state.get('metadata', {}).get('filters_applied', {})
                if filters_applied:
                    st.info(f"🔍 **Filters Applied:** Years {filters_applied['year_range'][0]}-{filters_applied['year_range'][1]} | " +
                           f"Study Types: {', '.join(filters_applied['study_types']) if filters_applied['study_types'] else 'All'} | " +
                           f"Min Quality: {filters_applied['min_quality']}")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Studies Analyzed", len(final_state['search_results']))
                with col2:
                    high_q = sum(1 for q in final_state.get('quality_scores', []) if q['quality_score'] >= 20)
                    st.metric("High Quality Studies", high_q)
                with col3:
                    avg_year = sum(int(s.get('year', 0)) for s in final_state['search_results']) / len(final_state['search_results']) if final_state['search_results'] else 0
                    st.metric("Average Year", f"{avg_year:.0f}")
                
                if final_state.get('error'):
                    st.warning(final_state['error'])
                
                # Quality distribution
                if final_state.get('quality_scores'):
                    st.markdown("### 📊 Evidence Quality Distribution")
                    
                    quality_data = {}
                    for q in final_state['quality_scores']:
                        grade = q['grade']
                        quality_data[grade] = quality_data.get(grade, 0) + 1
                    
                    cols = st.columns(len(quality_data))
                    for idx, (grade, count) in enumerate(quality_data.items()):
                        with cols[idx]:
                            st.metric(grade, count)
                
                st.markdown("---")
                
                # Main synthesis
                st.markdown("## 📄 Evidence Synthesis Report")
                st.markdown(final_state['final_summary'])
                
                # Detailed studies
                st.markdown("---")
                st.markdown("## 📚 Individual Study Details")
                
                for idx, doc in enumerate(final_state['search_results'], 1):
                    quality = next((q for q in final_state.get('quality_scores', []) if q['pmid'] == doc['pmid']), {})
                    
                    with st.expander(
                        f"📄 [{quality.get('quality_score', 0)}/{quality.get('max_score', 25)}] Study {idx}: {doc['title'][:100]}...", 
                        expanded=(idx == 1)
                    ):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("**Metadata:**")
                            st.write(f"**Year:** {doc['year']}")
                            st.write(f"**Journal:** {doc['journal']}")
                            st.write(f"**Sample Size:** {doc['sample_size']}")
                            st.write(f"**Citations:** {doc.get('citation_count', 0)}")
                            st.write(f"**PMID:** {doc['pmid']}")
                            st.write(f"**Quality:** {quality.get('grade', 'N/A')}")
                            
                            # Quality breakdown
                            st.markdown("**Quality Breakdown:**")
                            st.write(f"Design: {quality.get('design_score', 0)}/5")
                            st.write(f"Sample: {quality.get('sample_score', 0)}/5")
                            st.write(f"Recency: {quality.get('recency_score', 0)}/5")
                            st.write(f"Citations: {quality.get('citation_score', 0)}/5")
                            st.write(f"Impact: {quality.get('impact_score', 0)}/5")
                            
                            st.link_button("View on PubMed", doc['link'], use_container_width=True)
                        
                        with col2:
                            st.markdown("**Authors:**")
                            st.write(doc['authors'])
                            st.markdown("**Study Type:**")
                            st.write(', '.join(doc['publication_types'][:3]))
                        
                        st.markdown("**Abstract:**")
                        st.write(doc['abstract'])
                
                # Export
                st.markdown("---")
                export = f"""# Advanced Clinical Evidence Review
                
Query: {user_query}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Studies Analyzed: {len(final_state['search_results'])}
High Quality: {high_q}

Filters Applied:
- Year Range: {year_filter[0]}-{year_filter[1]}
- Study Types: {', '.join(study_types) if study_types else 'All'}
- Minimum Quality: {min_quality}

{final_state['final_summary']}

---
DETAILED STUDY LIST

"""
                for idx, doc in enumerate(final_state['search_results'], 1):
                    export += f"\n{idx}. {doc['title']}\n"
                    export += f"   {doc['authors']} ({doc['year']})\n"
                    export += f"   {doc['journal']}\n"
                    export += f"   Sample Size: {doc['sample_size']}\n"
                    export += f"   PMID: {doc['pmid']}\n\n"
                
                st.download_button(
                    "📥 Download Complete Report",
                    data=export,
                    file_name=f"evidence_review_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
        except Exception as e:
            progress.empty()
            status.empty()
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    print("="*70)
    print("🔬 Advanced Clinical Literature Synthesis Agent")
    print("="*70)
    print(f"Model: {Config.WORKING_MODEL}")
    print(f"Max Studies: {Config.MAX_RESULTS}")
    print("="*70)
    main()