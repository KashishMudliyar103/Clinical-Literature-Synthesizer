# 🔬 Clinical Literature Synthesizer

A LangGraph-based multi-agent system for automated clinical literature review, quality assessment, and evidence synthesis using PubMed and Gemini AI.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🌟 Features

- **Automated PubMed Search**: Retrieves clinical studies with advanced filtering
- **5-Dimensional Quality Assessment**: Evaluates studies on design, sample size, recency, citations, and journal impact
- **AI-Powered Synthesis**: Generates structured evidence reports with Gemini 2.5
- **Interactive Dashboard**: Streamlit UI with real-time filters and visualizations
- **Comparative Analysis**: Side-by-side study comparison with quality metrics
- **Export Functionality**: Download complete reports in markdown format

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Search Node    │ ◄── Filters: Year, Study Type
│  (PubMed API)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quality Node    │ ◄── 5 Criteria Assessment
│ (0-25 Scale)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Filter Node    │ ◄── Min Quality Threshold
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Synthesis Node  │ ◄── Gemini 2.5 AI
│ (Evidence Report)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Output   │
│ (Markdown Report)│
└─────────────────┘
```

## 📊 Quality Scoring Methodology

Each study receives a score from 0-25 based on five criteria:

| Criterion | Weight | Scale | Description |
|-----------|--------|-------|-------------|
| **Study Design** | 0-5 | RCT/Meta-analysis (5), Cohort (3), Case-control (2), Other (1) | Evidence hierarchy |
| **Sample Size** | 0-5 | ≥1000 (5), ≥500 (4), ≥100 (3), ≥50 (2), <50 (1) | Statistical power |
| **Recency** | 0-5 | ≤1yr (5), ≤2yr (4), ≤3yr (3), ≤5yr (2), >5yr (1) | Current relevance |
| **Citations** | 0-5 | ≥50/yr (5), ≥20/yr (4), ≥10/yr (3), ≥5/yr (2), >0 (1) | Impact factor |
| **Journal Tier** | 0-5 | Top (5), High-impact (4), Specialty (3), Peer-reviewed (2) | Publication quality |

**GRADE Classification:**
- 20-25: High Quality
- 14-19: Moderate Quality
- 8-13: Low Quality
- 0-7: Very Low Quality

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))
- Email for PubMed API access

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Clinical-Literature-Synthesizer.git
cd Clinical-Literature-Synthesizer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
# Copy example file
cp .env.example .env

# Edit .env file with your credentials
nano .env  # or use your preferred editor
```

Example `.env` configuration:
```env
GEMINI_API_KEY=AIzaSyD...your_key_here
ENTREZ_EMAIL=your_email@example.com
MAX_RESULTS=10
```

5. **Test your setup**
```bash
# Test Gemini API connection
python test_gemini.py

# Should output: "✅ SUCCESS! Everything is working!"
```

## 🎯 Usage

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Example Queries

Try these clinical questions:

- **Diabetes Management**: "Efficacy of metformin in type 2 diabetes"
- **Cardiovascular**: "SGLT2 inhibitors heart failure mortality"
- **Preventive Care**: "Statins primary prevention cardiovascular disease"
- **Weight Management**: "GLP-1 agonists weight loss diabetes"

### Using Filters

1. **Publication Years**: Limit search to specific date range (2020-2025)
2. **Study Types**: Filter by RCT, Meta-Analysis, Cohort, or Case-Control studies
3. **Minimum Quality**: Set quality threshold (Very Low, Low, Moderate, High)

### Interpreting Results

The system provides:

1. **Executive Summary**: Direct answer with evidence level
2. **Quality Distribution**: Visual breakdown of study quality
3. **Comparative Table**: Side-by-side study comparison
4. **Individual Studies**: Detailed abstracts with quality scores
5. **Clinical Implications**: Practical recommendations

## 📁 Project Structure

```
Clinical-Literature-Synthesizer/
├── app.py                    # Main Streamlit application
├── evaluate_system.py        # Evaluation framework
├── test_gemini.py           # API key testing utility
├── .env.example             # Environment template
├── .gitignore               # Git exclusions
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md               # This file
```

## 🔬 Evaluation System

The project includes a comprehensive evaluation framework:

```bash
python evaluate_system.py
```

**Evaluation Methods:**
1. **Structured (Trajectory) Evaluation**: Verifies agent workflow correctness
2. **LLM-as-Judge**: Quality assessment of generated reports

**Metrics Tracked:**
- Trajectory accuracy
- Element compliance (keywords, sections, tables)
- Clinical relevance
- Citation accuracy

## 🛠️ API Reference

### AgentState Structure

```python
class AgentState(TypedDict):
    user_query: str                    # Clinical question
    search_results: List[Dict]         # PubMed results
    quality_scores: List[Dict]         # Quality assessments
    final_summary: str                 # Generated report
    filters: Dict                      # User preferences
    error: str                         # Error messages
    metadata: Dict                     # Search metadata
```

### Workflow Nodes

- `search_node`: PubMed API integration with filters
- `quality_assessment_node`: 5-criteria scoring
- `filter_by_quality_node`: Threshold filtering
- `synthesis_node`: Gemini AI report generation

## 🔒 Security Best Practices

**NEVER commit API keys to Git!**

This project uses:
- `.env` files for secrets (excluded via `.gitignore`)
- `os.getenv()` for environment variable access
- Example files (`.env.example`) with placeholders only

If you accidentally commit a key:
1. Revoke it immediately on Google AI Studio
2. Generate a new key
3. Clean Git history with BFG Repo-Cleaner
4. Force push the cleaned repository

See [API Key Security Guide](docs/security_guide.md) for details.

## 🧪 Testing

### Manual Testing

```bash
# Test individual queries
python -c "from app import create_workflow; workflow = create_workflow(); print(workflow.invoke({'user_query': 'diabetes', 'filters': {}}))"
```

### Automated Evaluation

```bash
python evaluate_system.py
```

Expected output:
- Trajectory Accuracy: >95%
- Element Compliance: >85%
- LLM Judge Score: >0.80

## 📈 Performance

- **Average Query Time**: 8-12 seconds
- **Studies Analyzed**: 5-10 per query
- **Quality Assessment**: <1 second
- **AI Synthesis**: 3-5 seconds
- **Citation Fetching**: 2-4 seconds

## 🐛 Troubleshooting

### Common Issues

**"No results found"**
- Check query spelling
- Broaden search terms
- Adjust year filters
- Try different study type filters

**"API key invalid"**
- Verify key in `.env` file
- Ensure key starts with `AIza`
- Check Google AI Studio for key status
- Generate new key if needed

**"Rate limit exceeded"**
- PubMed allows 3 requests/second
- Wait 10 seconds and retry
- Consider reducing MAX_RESULTS

**"Empty synthesis"**
- Verify GEMINI_API_KEY is set
- Check test_gemini.py output
- Review Gemini API quotas

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PubMed/NCBI**: Clinical literature database
- **Google Gemini**: AI-powered synthesis
- **LangGraph**: Multi-agent orchestration framework
- **Streamlit**: Interactive web interface

## 📬 Contact

For questions or support:
- Open an issue on GitHub
- Email: kashishmudliar@gmail.com

---
