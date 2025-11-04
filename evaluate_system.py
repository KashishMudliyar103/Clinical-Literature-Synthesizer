# ================================================================
# LANGGRAPH EVALUATION FRAMEWORK
# Based on: End-to-End + Structured (Trajectory) Evaluation
# UPDATED: Fixed quality scoring to match app.py (5 criteria, 0-25 scale)
# ================================================================

import os
import json
import time
from typing import List, Dict, TypedDict
from datetime import datetime
from Bio import Entrez
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import re

load_dotenv()

# ================================================================
# CONFIGURATION
# ================================================================

class Config:
    ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    WORKING_MODEL = 'models/gemini-2.5-flash-preview-05-20'

genai.configure(api_key=Config.GEMINI_API_KEY)

# ================================================================
# STATE DEFINITION
# ================================================================

class AgentState(TypedDict):
    user_query: str
    search_results: List[Dict]
    quality_scores: List[Dict]
    final_summary: str
    error: str
    trajectory: List[str]  # Track agent actions

# ================================================================
# TEST CASES WITH EXPECTED BEHAVIOR
# ================================================================

TEST_CASES = [
    {
        "query": "Efficacy of metformin in type 2 diabetes",
        "domain": "Endocrinology",
        "expected_trajectory": [
            "search_node",
            "quality_assessment_node", 
            "synthesis_node"
        ],
        "expected_elements": {
            "must_include_keywords": ["metformin", "diabetes", "HbA1c", "glycemic"],
            "must_have_sections": ["Executive Summary", "Key Findings", "Clinical Implications", "Comparative Evidence Table"],
            "must_have_table_columns": ["Citations", "Impact"],
            "must_have_references": True,
            "min_studies": 5
        },
        "quality_criteria": {
            "is_markdown_formatted": True,
            "has_comparative_analysis": True,
            "cites_sources": True,
            "clinically_relevant": True
        }
    },
    {
        "query": "SGLT2 inhibitors heart failure",
        "domain": "Cardiology",
        "expected_trajectory": [
            "search_node",
            "quality_assessment_node",
            "synthesis_node"
        ],
        "expected_elements": {
            "must_include_keywords": ["SGLT2", "heart failure", "mortality", "hospitalization"],
            "must_have_sections": ["Executive Summary", "Evidence", "Recommendations", "Comparative Evidence Table"],
            "must_have_table_columns": ["Citations", "Impact"],
            "must_have_references": True,
            "min_studies": 5
        },
        "quality_criteria": {
            "is_markdown_formatted": True,
            "has_comparative_analysis": True,
            "cites_sources": True,
            "clinically_relevant": True
        }
    }
]

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

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


def estimate_journal_impact(journal_name: str) -> int:
    """Estimate journal impact tier (0-5)"""
    journal_name = journal_name.lower()
    
    # Tier 1: Top-tier journals
    tier1 = ['new england journal of medicine', 'nejm', 'lancet', 'jama',
             'nature', 'science', 'cell', 'nature medicine', 'bmj']
    
    # Tier 2: High-impact specialty
    tier2 = ['diabetes care', 'circulation', 'journal of clinical oncology',
             'annals of internal medicine', 'plos medicine', 'jama internal medicine',
             'european heart journal', 'diabetologia']
    
    # Check tiers
    if any(j in journal_name for j in tier1):
        return 5
    elif any(j in journal_name for j in tier2):
        return 4
    elif 'journal' in journal_name:
        return 3
    elif journal_name and journal_name != 'unknown journal':
        return 2
    
    return 0

# ================================================================
# SYSTEM NODES (Matching app.py exactly)
# ================================================================

def search_node(state: AgentState) -> AgentState:
    """Search PubMed with enhanced metadata (MATCHES app.py)"""
    state['trajectory'].append("search_node")
    
    try:
        Entrez.email = Config.ENTREZ_EMAIL
        handle = Entrez.esearch(
            db="pubmed",
            term=state['user_query'],
            retmax=10,
            sort='relevance',
            mindate="2020",
            maxdate="2025"
        )
        record = Entrez.read(handle)
        handle.close()
        
        pubmed_ids = record['IdList']
        
        if not pubmed_ids:
            state['error'] = "No results"
            return state
        
        # Fetch details
        handle = Entrez.efetch(db="pubmed", id=','.join(pubmed_ids), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        results = []
        for record in records['PubmedArticle']:
            try:
                article = record['MedlineCitation']['Article']
                pmid = str(record['MedlineCitation']['PMID'])
                
                abstract_parts = article.get('Abstract', {}).get('AbstractText', [])
                abstract = ' '.join([str(part) for part in abstract_parts]) if abstract_parts else "No abstract"
                
                # Get publication types
                pub_types = []
                for pub_type in article.get('PublicationTypeList', []):
                    pub_types.append(str(pub_type))
                
                # Get journal info
                journal_info = article.get('Journal', {})
                journal = journal_info.get('Title', 'Unknown Journal')
                year = journal_info.get('JournalIssue', {}).get('PubDate', {}).get('Year', 'Unknown')
                
                # Get citation count
                citation_count = 0
                try:
                    link_handle = Entrez.elink(dbfrom="pubmed", id=pmid, linkname="pubmed_pubmed_citedin")
                    link_record = Entrez.read(link_handle)
                    link_handle.close()
                    
                    if link_record and link_record[0]['LinkSetDb']:
                        citation_count = len(link_record[0]['LinkSetDb'][0]['Link'])
                except:
                    citation_count = 0
                
                results.append({
                    'pmid': pmid,
                    'title': str(article.get('ArticleTitle', '')),
                    'abstract': abstract[:2000],
                    'year': str(year),
                    'journal': journal,
                    'publication_types': pub_types,
                    'citation_count': citation_count
                })
            except:
                continue
        
        state['search_results'] = results
        
    except Exception as e:
        state['error'] = f"Search error: {e}"
    
    return state


def quality_assessment_node(state: AgentState) -> AgentState:
    """
    Enhanced quality assessment with 5 dimensions (0-25 scale)
    UPDATED: Now matches app.py exactly with all 5 criteria
    """
    state['trajectory'].append("quality_assessment_node")
    
    scores = []
    for study in state['search_results']:
        score_components = {}
        
        # 1. Study Design Score (0-5)
        pub_types = ' '.join(study.get('publication_types', [])).lower()
        if 'randomized controlled trial' in pub_types or 'clinical trial' in pub_types:
            score_components['design'] = 5
        elif 'meta-analysis' in pub_types or 'systematic review' in pub_types:
            score_components['design'] = 5
        elif 'cohort' in pub_types:
            score_components['design'] = 3
        elif 'case-control' in pub_types:
            score_components['design'] = 2
        else:
            score_components['design'] = 1
        
        # 2. Sample Size Score (0-5) - NOW INCLUDED!
        sample_str = extract_sample_size(study.get('abstract', ''))
        try:
            n = int(sample_str) if sample_str != "Not specified" else 0
            if n >= 1000:
                score_components['sample'] = 5
            elif n >= 500:
                score_components['sample'] = 4
            elif n >= 100:
                score_components['sample'] = 3
            elif n >= 50:
                score_components['sample'] = 2
            elif n > 0:
                score_components['sample'] = 1
            else:
                score_components['sample'] = 0
        except:
            score_components['sample'] = 0
        
        # 3. Recency Score (0-5)
        try:
            year = int(study.get('year', 0))
            age = 2025 - year
            if age <= 1:
                score_components['recency'] = 5
            elif age <= 2:
                score_components['recency'] = 4
            elif age <= 3:
                score_components['recency'] = 3
            elif age <= 5:
                score_components['recency'] = 2
            else:
                score_components['recency'] = 1
        except:
            score_components['recency'] = 0
        
        # 4. Citation Score (0-5) - Normalized by publication age
        citation_count = study.get('citation_count', 0)
        try:
            year = int(study.get('year', 2025))
            years_since_pub = max(1, 2025 - year)
            citations_per_year = citation_count / years_since_pub
            
            if citations_per_year >= 50:
                score_components['citations'] = 5
            elif citations_per_year >= 20:
                score_components['citations'] = 4
            elif citations_per_year >= 10:
                score_components['citations'] = 3
            elif citations_per_year >= 5:
                score_components['citations'] = 2
            elif citation_count > 0:
                score_components['citations'] = 1
            else:
                score_components['citations'] = 0
        except:
            score_components['citations'] = 0
        
        # 5. Journal Impact Score (0-5)
        impact_score = estimate_journal_impact(study.get('journal', ''))
        score_components['impact'] = impact_score
        
        # Calculate Total Score (0-25) - NO ARBITRARY ADDITIONS!
        total = sum(score_components.values())
        
        # GRADE Classification (matching app.py thresholds)
        if total >= 20:
            grade = "High Quality"
        elif total >= 14:
            grade = "Moderate Quality"
        elif total >= 8:
            grade = "Low Quality"
        else:
            grade = "Very Low Quality"
        
        scores.append({
            'pmid': study['pmid'],
            'score': total,
            'grade': grade,
            'design_score': score_components['design'],
            'sample_score': score_components['sample'],
            'recency_score': score_components['recency'],
            'citation_score': score_components['citations'],
            'impact_score': score_components['impact'],
            'max_score': 25
        })
    
    state['quality_scores'] = scores
    return state


def synthesis_node(state: AgentState) -> AgentState:
    """Generate synthesis (MATCHES app.py)"""
    state['trajectory'].append("synthesis_node")
    
    try:
        model = genai.GenerativeModel(Config.WORKING_MODEL)
        
        docs_text = ""
        for idx, doc in enumerate(state['search_results'][:5], 1):
            quality = next((q for q in state['quality_scores'] if q['pmid'] == doc['pmid']), {})
            
            docs_text += f"\n{'='*60}\n"
            docs_text += f"STUDY {idx}\n"
            docs_text += f"{'='*60}\n"
            docs_text += f"Title: {doc['title']}\n"
            docs_text += f"Journal: {doc['journal']} ({doc['year']})\n"
            docs_text += f"PMID: {doc['pmid']}\n"
            docs_text += f"Study Type: {', '.join(doc['publication_types'][:3])}\n"
            docs_text += f"Citations: {doc.get('citation_count', 0)}\n"
            docs_text += f"Quality Grade: {quality.get('grade', 'Not assessed')} ({quality.get('score', 0)}/25)\n"
            docs_text += f"\nABSTRACT:\n{doc['abstract'][:800]}\n"
        
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
[Fill each row with ACTUAL data from the studies above]

# KEY FINDINGS BY STUDY

[For EACH study provide:]
**Study [N]: [Short Title] ([Year])**
- Design & Population: [specifics]
- Intervention: [what was tested]
- Primary Outcome: [main result with numbers]
- Quality Assessment: [reference the grade provided]

# SYNTHESIS & CLINICAL IMPLICATIONS

**Consistency of Evidence:** [Do studies agree?]

**Strength of Recommendation:** [Based on quality and consistency]

**Clinical Application:** [Practical guidance]

**Knowledge Gaps:** [What's unclear?]

# REFERENCES
[List all studies with PMID links]

CRITICAL: Use ACTUAL numbers from studies, cite quality grades provided above."""

        response = model.generate_content(prompt)
        state['final_summary'] = response.text if response else "Generation failed"
        
    except Exception as e:
        state['error'] = f"Synthesis error: {e}"
        state['final_summary'] = "Error in synthesis"
    
    return state


# ================================================================
# EVALUATION 1: STRUCTURED (TRAJECTORY) EVALUATION
# ================================================================

class StructuredEvaluator:
    """Evaluate agent trajectory - did it take correct steps?"""
    
    @staticmethod
    def evaluate_trajectory(actual: List[str], expected: List[str]) -> Dict:
        """Check if agent followed expected path"""
        
        trajectory_match = actual == expected
        all_nodes_visited = all(node in actual for node in expected)
        unexpected_nodes = [node for node in actual if node not in expected]
        
        return {
            "trajectory_correct": trajectory_match,
            "all_expected_nodes_visited": all_nodes_visited,
            "unexpected_nodes": unexpected_nodes,
            "actual_trajectory": actual,
            "expected_trajectory": expected,
            "score": 1.0 if trajectory_match else 0.0
        }
    
    @staticmethod
    def evaluate_elements(state: AgentState, expected: Dict) -> Dict:
        """Check if output has expected elements"""
        
        results = {}
        final_text = state['final_summary'].lower()
        
        # Check keywords
        keywords_found = sum(1 for kw in expected['must_include_keywords'] 
                           if kw.lower() in final_text)
        results['keywords_score'] = keywords_found / len(expected['must_include_keywords'])
        
        # Check sections
        sections_found = sum(1 for section in expected['must_have_sections']
                           if section.lower() in final_text)
        results['sections_score'] = sections_found / len(expected['must_have_sections'])
        
        # Check table columns
        if 'must_have_table_columns' in expected:
            table_cols_found = sum(1 for col in expected['must_have_table_columns']
                                 if col.lower() in final_text)
            results['table_columns_score'] = table_cols_found / len(expected['must_have_table_columns'])
        else:
            results['table_columns_score'] = 1.0
        
        # Check references
        has_refs = 'pmid' in final_text or 'reference' in final_text
        results['has_references'] = has_refs
        
        # Check study count
        results['study_count'] = len(state['search_results'])
        results['meets_min_studies'] = len(state['search_results']) >= expected['min_studies']
        
        # Overall score
        results['overall_score'] = (
            results['keywords_score'] * 0.25 +
            results['sections_score'] * 0.25 +
            results['table_columns_score'] * 0.15 +
            (1.0 if has_refs else 0.0) * 0.20 +
            (1.0 if results['meets_min_studies'] else 0.0) * 0.15
        )
        
        return results


# ================================================================
# EVALUATION 2: END-TO-END (LLM AS JUDGE)
# ================================================================

class LLMJudge:
    """Use LLM to evaluate final output quality"""
    
    def __init__(self):
        self.model = genai.GenerativeModel(Config.WORKING_MODEL)
    
    def evaluate_output(self, query: str, output: str, criteria: Dict) -> Dict:
        """Have LLM judge the output quality"""
        
        judge_prompt = f"""You are evaluating a clinical literature synthesis system.

USER QUERY: {query}

SYSTEM OUTPUT:
{output}

EVALUATION CRITERIA:
1. Is it properly formatted in Markdown? (Yes/No)
2. Does it include comparative analysis table with Citations and Impact columns? (Yes/No)
3. Does it cite sources with PMIDs? (Yes/No)
4. Is the content clinically relevant and accurate? (Yes/No)
5. Does it provide actionable clinical recommendations? (Yes/No)

For each criterion, respond with:
- Score: 0 (No) or 1 (Yes)
- Justification: Brief explanation

Then provide:
- Overall Score: Average of all scores (0.0 to 1.0)
- Strengths: What the output does well
- Weaknesses: What needs improvement

Format your response as JSON:
{{
  "markdown_formatted": {{"score": 0 or 1, "justification": "..."}},
  "comparative_analysis": {{"score": 0 or 1, "justification": "..."}},
  "cites_sources": {{"score": 0 or 1, "justification": "..."}},
  "clinically_relevant": {{"score": 0 or 1, "justification": "..."}},
  "actionable_recommendations": {{"score": 0 or 1, "justification": "..."}},
  "overall_score": 0.0 to 1.0,
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."]
}}"""

        try:
            response = self.model.generate_content(judge_prompt)
            response_text = response.text
            
            # Extract JSON
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text
            
            result = json.loads(json_str)
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "overall_score": 0.0
            }


# ================================================================
# COMBINED EVALUATION RUNNER
# ================================================================

def run_evaluation():
    """Run both evaluation methods"""
    
    print("="*80)
    print("LANGGRAPH AGENT EVALUATION - UPDATED WITH CONSISTENT SCORING")
    print("="*80)
    
    # Create workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("quality_assessment", quality_assessment_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.set_entry_point("search")
    workflow.add_edge("search", "quality_assessment")
    workflow.add_edge("quality_assessment", "synthesis")
    workflow.add_edge("synthesis", END)
    app = workflow.compile()
    
    # Initialize evaluators
    structured_eval = StructuredEvaluator()
    llm_judge = LLMJudge()
    
    results = []
    
    for idx, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[Test {idx}/{len(TEST_CASES)}] {test_case['query']}")
        print("-"*80)
        
        # Run agent
        start_time = time.time()
        initial_state = {
            "user_query": test_case['query'],
            "search_results": [],
            "quality_scores": [],
            "final_summary": "",
            "error": "",
            "trajectory": []
        }
        
        final_state = app.invoke(initial_state)
        elapsed_time = time.time() - start_time
        
        # EVALUATION 1: Structured (Trajectory)
        print("\n1. STRUCTURED EVALUATION (Trajectory)")
        traj_eval = structured_eval.evaluate_trajectory(
            final_state['trajectory'],
            test_case['expected_trajectory']
        )
        print(f"   Trajectory Match: {'✓' if traj_eval['trajectory_correct'] else '✗'}")
        print(f"   Score: {traj_eval['score']:.2f}")
        print(f"   Time: {elapsed_time:.2f}s")
        
        element_eval = structured_eval.evaluate_elements(
            final_state,
            test_case['expected_elements']
        )
        print(f"\n   Elements Evaluation:")
        print(f"   - Keywords: {element_eval['keywords_score']:.2f}")
        print(f"   - Sections: {element_eval['sections_score']:.2f}")
        print(f"   - Table Columns: {element_eval['table_columns_score']:.2f}")
        print(f"   - References: {'✓' if element_eval['has_references'] else '✗'}")
        print(f"   - Study Count: {element_eval['study_count']} (min: {test_case['expected_elements']['min_studies']})")
        print(f"   - Overall: {element_eval['overall_score']:.2f}")
        
        # EVALUATION 2: LLM Judge
        print("\n2. END-TO-END EVALUATION (LLM Judge)")
        judge_eval = llm_judge.evaluate_output(
            test_case['query'],
            final_state['final_summary'],
            test_case['quality_criteria']
        )
        
        if 'error' not in judge_eval:
            print(f"   Overall Judge Score: {judge_eval['overall_score']:.2f}")
            print(f"   Strengths: {', '.join(judge_eval.get('strengths', ['N/A']))}")
            print(f"   Weaknesses: {', '.join(judge_eval.get('weaknesses', ['N/A']))}")
        else:
            print(f"   Judge Error: {judge_eval['error']}")
        
        # Quality Score Analysis
        print(f"\n3. QUALITY ASSESSMENT ANALYSIS")
        if final_state['quality_scores']:
            scores = [q['score'] for q in final_state['quality_scores']]
            print(f"   Average Quality: {sum(scores)/len(scores):.1f}/25")
            print(f"   Score Range: {min(scores)}-{max(scores)}")
            
            # Grade distribution
            grades = {}
            for q in final_state['quality_scores']:
                grade = q['grade']
                grades[grade] = grades.get(grade, 0) + 1
            print(f"   Grade Distribution: {grades}")
        
        # Store results
        results.append({
            "test_case": test_case['query'],
            "domain": test_case['domain'],
            "elapsed_time": elapsed_time,
            "trajectory_evaluation": traj_eval,
            "element_evaluation": element_eval,
            "judge_evaluation": judge_eval,
            "quality_stats": {
                "scores": [q['score'] for q in final_state['quality_scores']],
                "grades": [q['grade'] for q in final_state['quality_scores']]
            }
        })
    
    # ================================================================
    # AGGREGATE RESULTS
    # ================================================================
    
    print("\n" + "="*80)
    print("AGGREGATE RESULTS")
    print("="*80)
    
    # Trajectory accuracy
    traj_scores = [r['trajectory_evaluation']['score'] for r in results]
    print(f"\nTrajectory Accuracy: {sum(traj_scores)/len(traj_scores):.2%}")
    
    # Element scores
    element_scores = [r['element_evaluation']['overall_score'] for r in results]
    print(f"Element Compliance: {sum(element_scores)/len(element_scores):.2%}")
    
    # Judge scores
    judge_scores = [r['judge_evaluation'].get('overall_score', 0) for r in results]
    print(f"LLM Judge Score: {sum(judge_scores)/len(judge_scores):.2%}")
    
    # Average time
    times = [r['elapsed_time'] for r in results]
    print(f"Average Execution Time: {sum(times)/len(times):.2f}s")
    
    # Quality statistics
    print(f"\n--- Quality Assessment Statistics ---")
    all_scores = []
    for r in results:
        all_scores.extend(r['quality_stats']['scores'])
    
    if all_scores:
        print(f"Overall Average Quality: {sum(all_scores)/len(all_scores):.1f}/25")
        print(f"Quality Range: {min(all_scores)}-{max(all_scores)}")
        
        # Count by grade
        all_grades = []
        for r in results:
            all_grades.extend(r['quality_stats']['grades'])
        
        from collections import Counter
        grade_counts = Counter(all_grades)
        print(f"Grade Distribution Across All Tests:")
        for grade, count in sorted(grade_counts.items()):
            print(f"  {grade}: {count} ({count/len(all_grades)*100:.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"langgraph_evaluation_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_file}")
    
    return results


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("\nLangGraph Agent Evaluation Framework")
    print("UPDATED: Quality scoring now consistent with app.py (5 criteria, 0-25 scale)")
    print("Methods: 1) Structured (Trajectory) + 2) LLM-as-Judge\n")
    
    results = run_evaluation()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)