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