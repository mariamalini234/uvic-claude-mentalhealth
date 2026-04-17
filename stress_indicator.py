
# -------------------------
# A hierarchical stress inference + decision + AI conditioning system, followed by prompt injection through HTML.
# SBERT Stress Score Pipeline per User - For AI reinforcement learning, session history creation
# PIPELINE 1 (BASELINE): Conversation-level stress estimation using full dialogue context (input + output)
# -------------------------

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os


# --- Step 1: Load both input files
stressed_file = './data/Categorized_Mental_Health_Data.csv'
normal_file = './data/Dummy_Mental_Health_Conversations.csv'

df_stress = pd.read_csv(stressed_file)
df_normal = pd.read_csv(normal_file)

# --- Step 2: Add user_type column if missing
df_stress['user_type'] = 'stressed'
df_normal['user_type'] = 'normal'

# --- Step 3: Combine both datasets
df = pd.concat([df_stress, df_normal], ignore_index=True)

# --- Step 4: Clean column names
df.columns = df.columns.str.strip()

# --- Step 5: Group by user (average their messages)
# First, combine messages per user by concatenating input + output
df['combined_text'] = df['input'].fillna('') + '. ' + df['output'].fillna('')

# Now group by user
df_user = df.groupby('user').agg({
    'combined_text': lambda x: ' '.join(x),
    'user_type': 'first'
}).reset_index()

# --- Step 6: Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Step 7: Encode each user's combined messages
embeddings = model.encode(df_user['combined_text'].tolist(), show_progress_bar=True)

# --- Step 8: Create SBERT anchor for stress
anchors = {"stressed": model.encode([
    "I feel stressed, anxious, overwhelmed, worried, pressured, exhausted"
])}

# --- Step 9: Compute SBERT stress score per user (0-100 integer)
def sbert_stress_score(row_emb, anchor_emb):
    sim = cosine_similarity(row_emb.reshape(1,-1), anchor_emb)[0][0]
    return int(round(sim * 100))

df_user['stress_score'] = [sbert_stress_score(embeddings[i], anchors['stressed'])
                           for i in range(len(df_user))]

# --- Step 10: Save per-user SBERT stress scores, this is per user stress score
project_path = './output'
os.makedirs(project_path, exist_ok=True)
output_file = os.path.join(project_path, "sbert_stress_scores_per_user.csv")
df_user.to_csv(output_file, index=False)

print("SBERT stress scores per user saved at:", output_file)

# --- Step 11: Quick verification
print("\nAverage SBERT stress per user_type:")
print(df_user.groupby('user_type')['stress_score'].mean())

# Optional: preview
df_user.head(10)



# Paths
stressed_file = './data/Categorized_Mental_Health_Data.csv'
normal_file = './data/Dummy_Mental_Health_Conversations.csv'

# Load datasets
df_stress = pd.read_csv(stressed_file)
df_normal = pd.read_csv(normal_file)

# Add user_type column
df_stress['user_type'] = 'stressed'
df_normal['user_type'] = 'normal'

# --- Step 1a: Ensure unique user IDs for normal dataset ---
df_normal['user'] = df_normal['user'].apply(lambda x: f"normal_{x}")

# Optional: preview first 5 rows
print("Stressed users:")
print(df_stress[['user','user_type']].head())
print("\nNormal users:")
print(df_normal[['user','user_type']].head())



# --- Step 2: Combine stressed and normal datasets
df_combined = pd.concat([df_stress, df_normal], ignore_index=True)

# Optional: verify number of unique users
print("Unique users in combined data:", df_combined['user'].nunique())
print("Number of stressed users:", df_combined[df_combined['user_type']=='stressed']['user'].nunique())
print("Number of normal users:", df_combined[df_combined['user_type']=='normal']['user'].nunique())

# --- Step 2a: Save combined data
project_path = './output'
os.makedirs(project_path, exist_ok=True)
combined_file = os.path.join(project_path, "combined_mentalhealth_data.csv")
df_combined.to_csv(combined_file, index=False)

print("Combined dataset saved at:", combined_file)

# PIPELINE 2 (PRIMARY MODEL): User stress estimation using only user inputs aggregated across sessions
# Use this

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Step 1: Stress anchors
stress_anchors = [
    "I feel overwhelmed with too many responsibilities and pressure",
    "I feel anxious, worried all the time and cannot relax",
    "I feel mentally exhausted and unable to cope anymore",
    "I feel like I am failing and nothing is working",
    "I feel emotionally drained and stuck in my situation"
]

# --- Step 2: Calm / normal anchors
calm_anchors = [
    "I feel calm, relaxed and at peace",
    "I am doing fine and managing things well",
    "I feel happy, stable and in control of my life",
    "Everything is okay and I feel balanced",
    "I feel content and not stressed"
]

# Encode anchors
stress_emb = model.encode(stress_anchors)
calm_emb = model.encode(calm_anchors)

# --- Step 3: Dual scoring function
#def compute_dual_stress(row_emb, stress_emb, calm_emb):
 #   stress_sim = max(cosine_similarity(row_emb.reshape(1,-1), stress_emb)[0])
  #  calm_sim = max(cosine_similarity(row_emb.reshape(1,-1), calm_emb)[0])

   # score = stress_sim - calm_sim   # range approx -1 to +1

    # scale to 0–100
    #scaled = int(round((score + 1) / 2 * 100))
    #return scaled

# --- Step 3: relative scoring function

def compute_dual_stress(row_emb, stress_emb, calm_emb):
    stress_sim = max(cosine_similarity(row_emb.reshape(1,-1), stress_emb)[0])
    calm_sim = max(cosine_similarity(row_emb.reshape(1,-1), calm_emb)[0])

    # ratio-based score
    ratio = stress_sim / (stress_sim + calm_sim + 1e-6)

    return int(round(ratio * 100))


# --- Step 4: Apply to all rows (input embeddings)
#df_combined['sbert_stress_score'] = [
 #   compute_dual_stress(input_embeddings[i], stress_emb, calm_emb)
 #   for i in range(len(df_combined))
#]

input_embeddings = model.encode(
    df_combined['input'].fillna('').tolist(),
    show_progress_bar=True
)

df_combined['sbert_stress_score'] = [
    compute_dual_stress(emb, stress_emb, calm_emb)
    for emb in input_embeddings
]

# --- Step 4: Apply to all rows (input embeddings) - different model.
df_combined["stress_score_scaled"] = df_combined["sbert_stress_score"].apply(
    lambda x: max(0, min(100, int((x - 50) * 2 + 50)))
)

# --- Step 5: Save
combined_file = './output/combined_mentalhealth_data1.csv'
df_combined.to_csv(combined_file, index=False)

print("Dual-anchor SBERT stress scores saved.")

# --- Step 6: Check separation
print("\nAverage SBERT stress by user_type:")
print(df_combined.groupby("user_type")["sbert_stress_score"].mean())


df_user_avg = df_combined.groupby("user").agg({
    "stress_score_scaled": "mean",
    "user_type": "first"   # keep their label
}).reset_index()

# Quick check
print(df_user_avg.head())

def stress_color(score):
    if score < 45:
        return "green"   # Calm
    elif score < 55:
        return "yellow"  # Mild
    elif score < 70:
        return "orange"  # Moderate
    else:
        return "red"     # High Stress

df_user_avg["stress_color"] = df_user_avg["stress_score_scaled"].apply(stress_color)

def stress_label(score):
    if score < 45:
        return "Calm"
    elif score < 55:
        return "Mild"
    elif score < 70:
        return "Moderate"
    else:
        return "High"

df_user_avg["stress_label"] = df_user_avg["stress_score_scaled"].apply(stress_label)

output_path = './output/user_stress_summary.csv'
df_user_avg.to_csv(output_path, index=False)
print(f"User-level stress summary saved at: {output_path}")

# (LLM CONTEXT LAYER): Compress user history + stress signals for downstream AI response generation

# Combine messages into a single text per user
df_session_history = df_combined.groupby("user").agg({
    "input": lambda texts: "\n".join(texts),  # all user messages in order
    "stress_score_scaled": "mean",
    "user_type": "first"
}).reset_index()

def generate_AI_prompt_with_history(row):
    return f"""User: {row['user']}
Type: {row['user_type']}
Average Stress Score: {round(row['stress_score_scaled'],1)}/100
Stress Level: {"Calm" if row['stress_score_scaled']<45 else "Mild" if row['stress_score_scaled']<55 else "Moderate/High"}
Color Indicator: {"green" if row['stress_score_scaled']<45 else "yellow" if row['stress_score_scaled']<55 else "orange/red"}
Session History (User Messages):
{row['input']}

Summary: Based on the above messages and stress score, provide insights, suggestions, or next steps for this user."""


df_session_history['AI_prompt'] = df_session_history.apply(generate_AI_prompt_with_history, axis=1)

# Save to CSV
output_path = "./output/AI_user_prompts_with_history.csv"
df_session_history.to_csv(output_path, index=False, columns=['user','AI_prompt'])
print(f"AI-ready prompts with session history saved at: {output_path}")
