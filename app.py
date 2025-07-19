import streamlit as st
import joblib
import numpy as np
import re

# ------------------------- Page Config -------------------------
st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="🛡️",
    layout="centered"
)

# ------------------------- Load Model -------------------------
@st.cache_resource
def load_model():
    return joblib.load("sgd_pipeline_model.pkl")

model = load_model()

# ------------------------- Replacement Dictionary -------------------------
replacement_dict = {
    "kill": "harm", "hate": "dislike", "stupid": "uninformed",
    "idiot": "misguided person", "dumb": "misinformed",
    "terrorist": "aggressor", "bomb": "threat", "racist": "biased",
    "slur": "offensive word", "fool": "silly person", "crazy": "irrational",
    "mad": "angry", "nonsense": "unhelpful", "jerk": "rude person",
    "trash": "not valuable", "moron": "ill-informed person",
    "violence": "conflict", "abuse": "mistreatment", "pig": "impolite person",
    "suck": "is not ideal", "loser": "less successful person",
    "ugly": "less attractive", "retard": "person with delay"
}

# ------------------------- Utility Functions -------------------------
def modify_text(text, replacements):
    found_words = []
    def replace(match):
        word = match.group(0)
        replacement = replacements.get(word.lower(), word)
        if word.lower() in replacements:
            found_words.append(word)
        return replacement
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in replacements) + r')\b', flags=re.IGNORECASE)
    modified = pattern.sub(replace, text)
    return modified, found_words

def get_hate_and_not_hate_words(pipeline, text):
    vect = pipeline.named_steps['vect']
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    feature_names = vect.get_feature_names_out()
    X = vect.transform([text])
    X_tfidf = tfidf.transform(X)
    hate_words = []
    not_hate_words = []
    if hasattr(clf, 'coef_'):
        coefs = clf.coef_[0]
        word_contributions = X_tfidf.toarray()[0] * coefs
        for i, contrib in enumerate(word_contributions):
            if contrib > 0:
                hate_words.append(feature_names[i])
            elif contrib < 0:
                not_hate_words.append(feature_names[i])
    return hate_words, not_hate_words

def label_to_text(label):
    return "🟥 Hate Speech" if label == 1 else "✅ Not Hate Speech"

def split_sentences(text):
    # Split by period or newline
    lines = re.split(r'[\n\.]+', text)
    return [line.strip() for line in lines if line.strip()]

# ------------------------- Streamlit UI -------------------------
st.title("🛡️ Hate Speech Detector")
st.markdown("Enter one or more sentences to analyze for hate speech, see predictions, and view softened versions.")

user_input = st.text_area("📝 Input your sentence(s) below (use periods or newlines to separate):")

# ------------------------- Prediction Button -------------------------
if st.button("🔍 Predict"):
    if user_input.strip():
        sentences = split_sentences(user_input)
        st.markdown(f"### 🧾 Total Sentences Detected: `{len(sentences)}`")

        for idx, sentence in enumerate(sentences, 1):
            st.divider()
            st.markdown(f"#### 🔢 Sentence {idx}")
            st.markdown(f"**Original:** {sentence}")

            prediction = model.predict([sentence])[0]
            prob = model.predict_proba([sentence])[0][1]
            hate_words, not_hate_words = get_hate_and_not_hate_words(model, sentence)
            modified_text, replaced = modify_text(sentence, replacement_dict)

            st.markdown(f"**Prediction:** {label_to_text(prediction)}")
            st.markdown(f"**📊 Probability of Hate Speech:** `{prob:.2f}`")
            st.markdown(f"**🔥 Hate Words:** {', '.join(hate_words) if hate_words else '🚫 None'}")
            st.markdown(f"**🕊️ Non-Hate Words:** {', '.join(not_hate_words) if not_hate_words else '🚫 None'}")

            st.markdown("**✨ Softened Version:**")
            st.code(modified_text, language='markdown')
            if replaced:
                st.markdown(f"**🔁 Words Replaced:** {', '.join(set(replaced))}")
            else:
                st.markdown("✅ No harsh words detected in dictionary.")
    else:
        st.warning("⚠️ Please enter some text.")
