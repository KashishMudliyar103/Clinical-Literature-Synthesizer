# ================================================================
# LANGGRAPH EVALUATION FRAMEWORK
# Based on: End-to-End + Structured (Trajectory) Evaluation
# ================================================================

import os
import json
from typing import List, Dict, TypedDict
from datetime import datetime
from Bio import Entrez
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

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
            "must_have_table_columns": ["Citations", "Impact"],  # New requirement
            "must_have_references": True,
            "min_studies": 5,
            "max_response_time": 15.0  # Increased due to citation fetching
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
            "must_have_table_columns": ["Citations", "Impact"],  # New requirement
            "must_have_references": True,
            "min_studies": 5,
            "max_response_time": 15.0  # Increased due to citation fetching
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
# SYSTEM NODES (Your Actual System)
# ================================================================

def search_node(state: AgentState) -> AgentState:
    """Search PubMed with enhanced metadata"""
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
                
                # Get journal
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
    """Enhanced quality assessment with 5 dimensions"""
    state['trajectory'].append("quality_assessment_node")
    
    scores = []
    for study in state['search_results']:
        # Design score (0-5)
        design_score = 0
        pub_types = ' '.join(study.get('publication_types', [])).lower()
        if 'randomized controlled trial' in pub_types or 'clinical trial' in pub_types:
            design_score = 5
        elif 'meta-analysis' in pub_types or 'systematic review' in pub_types:
            design_score = 5
        elif 'cohort' in pub_types:
            design_score = 3
        else:
            design_score = 1
        
        # Recency score (0-5)
        recency_score = 0
        try:
            year = int(study.get('year', 0))
            age = 2025 - year
            if age <= 1:
                recency_score = 5
            elif age <= 2:
                recency_score = 4
            elif age <= 3:
                recency_score = 3
            elif age <= 5:
                recency_score = 2
            else:
                recency_score = 1
        except:
            recency_score = 0
        
        # Citation score (0-5)
        citation_score = 0
        citation_count = study.get('citation_count', 0)
        try:
            year = int(study.get('year', 2025))
            years_since_pub = max(1, 2025 - year)
            citations_per_year = citation_count / years_since_pub
            
            if citations_per_year >= 50:
                citation_score = 5
            elif citations_per_year >= 20:
                citation_score = 4
            elif citations_per_year >= 10:
                citation_score = 3
            elif citations_per_year >= 5:
                citation_score = 2
            elif citation_count > 0:
                citation_score = 1
        except:
            citation_score = 0
        
        # Impact score (0-5)
        impact_score = 0
        journal = study.get('journal', '').lower()
        if any(j in journal for j in ['new england journal', 'nejm', 'lancet', 'jama', 'nature', 'science', 'bmj']):
            impact_score = 5
        elif any(j in journal for j in ['diabetes care', 'circulation', 'annals internal']):
            impact_score = 4
        elif 'journal' in journal:
            impact_score = 3
        elif journal and journal != 'unknown journal':
            impact_score = 2
        
        # Total score (0-25, excluding sample size for simplicity)
        total = design_score + recency_score + citation_score + impact_score + 3  # Add 3 as default sample score
        
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
            'design_score': design_score,
            'recency_score': recency_score,
            'citation_score': citation_score,
            'impact_score': impact_score
        })
    
    state['quality_scores'] = scores
    return state


def synthesis_node(state: AgentState) -> AgentState:
    """Generate synthesis"""
    state['trajectory'].append("synthesis_node")
    
    try:
        model = genai.GenerativeModel(Config.WORKING_MODEL)
        
        docs_text = ""
        for idx, doc in enumerate(state['search_results'][:5], 1):
            docs_text += f"\nStudy {idx}: {doc['title']}\nAbstract: {doc['abstract'][:800]}\n"
        
        prompt = f"""Clinical Question: {state['user_query']}

Studies:
{docs_text}

Generate a structured clinical evidence synthesis with these sections:

## Executive Summary
[2-3 sentences answering the question]

## Key Findings
[Bullet points of main findings from studies]

## Evidence Quality
[Brief assessment]

## Clinical Implications
[Practical recommendations]

## References
[List studies with PMIDs]

Use markdown formatting and cite sources."""

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
        
        # Check exact sequence
        trajectory_match = actual == expected
        
        # Check if all expected nodes were visited
        all_nodes_visited = all(node in actual for node in expected)
        
        # Check for unexpected nodes
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
        
        # Check table columns (if specified)
        if 'must_have_table_columns' in expected:
            table_cols_found = sum(1 for col in expected['must_have_table_columns']
                                 if col.lower() in final_text)
            results['table_columns_score'] = table_cols_found / len(expected['must_have_table_columns'])
        else:
            results['table_columns_score'] = 1.0  # Not required, give full credit
        
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
            
            # Parse JSON from response
            response_text = response.text
            
            # Extract JSON (it might be wrapped in ```json```)
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
    print("LANGGRAPH AGENT EVALUATION")
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
        initial_state = {
            "user_query": test_case['query'],
            "search_results": [],
            "quality_scores": [],
            "final_summary": "",
            "error": "",
            "trajectory": []
        }
        
        final_state = app.invoke(initial_state)
        
        # EVALUATION 1: Structured (Trajectory)
        print("\n1. STRUCTURED EVALUATION (Trajectory)")
        traj_eval = structured_eval.evaluate_trajectory(
            final_state['trajectory'],
            test_case['expected_trajectory']
        )
        print(f"   Trajectory Match: {'✓' if traj_eval['trajectory_correct'] else '✗'}")
        print(f"   Score: {traj_eval['score']:.2f}")
        
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
        
        # Store results
        results.append({
            "test_case": test_case['query'],
            "domain": test_case['domain'],
            "trajectory_evaluation": traj_eval,
            "element_evaluation": element_eval,
            "judge_evaluation": judge_eval
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
    
    # Quality assessment statistics
    print(f"\n--- Quality Assessment Analysis ---")
    for idx, result in enumerate(results, 1):
        print(f"\nQuery {idx}: {result['test_case']}")
        if 'quality_scores' in results[idx-1]:
            scores = [q['score'] for q in results[idx-1].get('quality_scores', [])]
            if scores:
                print(f"  Average Quality Score: {sum(scores)/len(scores):.1f}/25")
                print(f"  Score Range: {min(scores)}-{max(scores)}")
    
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
    print("Methods: 1) Structured (Trajectory) + 2) LLM-as-Judge\n")
    
    results = run_evaluation()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)