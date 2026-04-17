MindSpace: A hierarchical stress inference + decision + AI conditioning system (SBERT-Based Stress Scoring Pipeline)

Overview: This project uses Sentence-BERT embeddings to estimate user stress levels from therapy conversation data. It includes two pipelines:

Pipeline 1: Computes session-level stress score per user based on combined messages
Pipeline 2: Aggregates all sessions per user to estimate overall stress state and normalize scores

Stress is computed using semantic similarity between user text and predefined stress/calm anchor sentences.

How to Run:
1. Install dependencies - pip install pandas numpy scikit-learn sentence-transformers
2. Prepare data -
Place datasets in: ./data/
Required files: Categorized_Mental_Health_Data.csv & Dummy_Mental_Health_Conversations.csv
3. Run pipeline - python stress_indicator.py
4. Outputs - Generated files will be saved in: ./output/
Includes: Per-user stress scores (session-level)
Combined dataset with dual-anchor scoring
User-level aggregated stress summary
AI-ready session history prompts
5. Core Method: SBERT model: all-MiniLM-L6-v2
Stress score = cosine similarity with stress anchors
Calm comparison used for relative scaling
Final scores normalized to 0–100 scale
6. Output Artifacts: sbert_stress_scores_per_user.csv → session-level stress
combined_mentalhealth_data*.csv → merged dataset
user_stress_summary.csv → user-level classification
AI_user_prompts_with_history.csv → LLM-ready context to be used in index.html
7. Next Steps: Improve anchor calibration (reduce bias toward neutral text)
Optional visualization of stress trends per user
Self-check webapp.




Mental Health Stress Analysis using SBERT:

Overview

This project uses SBERT embeddings to analyze user messages and estimate stress levels of user over a period to capture behavioral stress score. It detects calm vs stressed users, aggregates user-level stress scores, and generates structured prompts for AI agents (Claude) for personalized insights.

Features
Compute stress vs calm embeddings per user message
Aggregate messages to user-level average stress score (0–100)
Assign stress labels: Mild / Moderate / High
Color-coded output for dashboards: Green / Yellow / Orange / Red
Generate Claude-ready prompts including recent key messages

Example Output

User-Level Summary:

user	stress_score_scaled	user_type	stress_color	stress_label
normal_User 1	51	normal	yellow	Mild
User 13	70	stressed	red	High

Claude Prompt Example:

User details:
User: User 13
Type: Stressed
Average Stress Score: 70/100
Stress Level: High
Color Indicator: Red
Recent Messages:
- "I feel exhausted and under pressure at work."
- "I can’t sleep because of all the deadlines."
Summary: Provide insights or suggestions based on stress level and recent messages.

Prompt:
You are a supportive AI companion.

You will receive: a stress score, stress level, color indicator, session history, and the user's current message.

Reason internally about who this person is, what shifted, what they need right now, and what would be wrong to say. Never display your reasoning process. Never output any thinking tags or reasoning steps.

Your response should be only what the user sees — warm, human, grounded in what you observed. No bullet points. No lists. No visible analysis.

When trauma, avoidance patterns, or escalating distress are present in the intake data, weave professional support naturally into your first response.

Project team consists of 2 people.
1. Maria Malini Anthony
2. Felipe Motta Franco
