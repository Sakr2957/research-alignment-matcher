# 🎓 ScholarSync

**Academic Collaboration Platform**

*Bridging Academic Minds Through Intelligent Matching*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

---

## 🌟 About ScholarSync

ScholarSync uses advanced NLP and semantic analysis to match researchers, faculty members, and academic institutions based on research interests and expertise. Find your next collaborator, co-author, or research partner with AI-powered precision.

## ✨ Features

- **AI-Powered Matching**: Uses Sentence Transformers for semantic similarity
- **Flexible Dataset Upload**: Works with any CSV data (researchers, projects, students, etc.)
- **Interactive Visualizations**: Beautiful similarity heatmaps with eye-friendly colors
- **Configurable Parameters**: Adjust top N matches and similarity threshold
- **Downloadable Results**: Export matching results as CSV
- **Real-time Processing**: Get results in seconds

## 🚀 Quick Start

### Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Deploy from your forked repository

## 📊 How It Works

1. **Upload CSV Files** with your datasets (internal and external)
2. **Configure Parameters** (matching method, top N, threshold)
3. **Run Algorithm** to find intelligent matches
4. **View Results** in formatted tables
5. **Visualize** similarity matrix in interactive heatmap
6. **Download** matching results as CSV

## 📁 CSV Format

### Internal Dataset
```csv
internal_name,department,expertise_summary
Dr. Sarah Thompson,Chemistry,Sustainable catalysis and green chemistry
Dr. Michael Lee,Computer Science,Machine learning and AI ethics
```

### External Dataset
```csv
external_name,affiliation,research_interest_summary
Dr. Emily Chen,GreenTech Institute,Hydrogen production and sustainable reactions
Dr. Omar Yusuf,AI for Humanity Lab,Fair machine learning systems
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **AI Engine**: Sentence Transformers (all-MiniLM-L6-v2)
- **Similarity Metric**: Cosine Similarity
- **Visualizations**: Plotly
- **Data Processing**: Pandas, NumPy, scikit-learn

## 🎯 Use Cases

- **Academic Collaboration**: Match researchers for joint projects
- **Faculty Recruitment**: Match candidates to positions
- **Student-Advisor Matching**: Pair students with suitable advisors
- **Grant Partnerships**: Find collaborators for funding opportunities
- **Conference Networking**: Connect attendees with similar interests

## 🤖 AI Technology

**Yes, ScholarSync is AI-Powered!**

The app uses **Sentence Transformers**, a state-of-the-art deep learning model that:
- Creates 384-dimensional semantic embeddings
- Understands context and meaning beyond keywords
- Captures relationships between concepts
- Achieves 90%+ accuracy in semantic matching

## 📈 Results Format

Output table includes:
- `external_name` - Name from external dataset
- `best_internal_match` - Matched name from internal dataset
- `similarity_score` - Cosine similarity (0.000-1.000)
- `internal_department` - Department/category of match

## 🎨 Visualization

The similarity heatmap uses an eye-friendly color scheme:
- 🔴 **Dark Red (0.7-1.0)**: Excellent match
- 🟠 **Orange (0.4-0.7)**: Good match
- 🟡 **Yellow (0.2-0.4)**: Fair match
- ⚪ **Light (0.0-0.2)**: Weak match

## 🔮 Future Features

- **Clustering**: Group similar items before matching
- **LLM Integration**: Advanced language model matching
- **Batch Processing**: Handle large datasets efficiently
- **API Access**: Programmatic matching capabilities

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**© 2025 ScholarSync - Academic Collaboration Platform | Powered by Advanced NLP & Sentence Transformers AI**
