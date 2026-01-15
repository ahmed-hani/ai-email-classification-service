from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.utils.embedding_utils import embedding_predict

# تحميل الموديل مرة واحدة
model = joblib.load("app/models/email_classifier.pkl")
vectorizer = joblib.load("app/models/tfidf_vectorizer.pkl")
priority_encoder = joblib.load("app/models/priority_encoder.pkl")
privacy_encoder = joblib.load("app/models/privacy_encoder.pkl")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

REFERENCE_PRIORITY = {
    "HIGH": [
        "urgent action required",
        "immediate response needed",
        "critical issue"
    ],
    "MEDIUM": [
        "please review",
        "needs attention",
        "important but not urgent"
    ],
    "LOW": [
        "no urgency",
        "whenever you have time",
        "for your information"
    ]
}

PRIVACY_REF = {
    "HIGH": [
        "confidential ",
        "internal only",
        "do not share",
        "private data"
    ],
    "MEDIUM": [
        "internal use",
        "company discussion",
        "team update"
    ],
    "LOW": [
        "public announcement",
        "can be shared",
        "external communication"
    ]
}


PRIORITY_EMBEDDINGS = {
    label: embedding_model.encode(texts)
    for label, texts in REFERENCE_PRIORITY.items()
}
PRIVACY_EMBEDDINGS = {
    k: embedding_model.encode(v)
    for k, v in PRIVACY_REF.items()
}

app = FastAPI(title="Email AI Classifier")

# -------- Request Schema --------
class EmailRequest(BaseModel):
    subject: str
    content: str

# -------- Response Schema --------
class EmailResponse(BaseModel):
    priority: str
    priority_confidence: float
    privacy: str
    privacy_confidence: float
    source: str


@app.post("/classify-email", response_model=EmailResponse)
def classify_email(email: EmailRequest):

    text = email.subject + " " + email.content
    vector = vectorizer.transform([text])

    # -------- TF-IDF Prediction --------
    pred = model.predict(vector)
    probs = model.predict_proba(vector)

    # Priority
    p_idx = pred[0][0]
    priority = priority_encoder.inverse_transform([p_idx])[0]
    priority_conf = float(probs[0][0].max())

    # Privacy
    pr_idx = pred[0][1]
    privacy = privacy_encoder.inverse_transform([pr_idx])[0]
    privacy_conf = float(probs[1][0].max())

    # -------- Decision Logic --------
    THRESHOLD = 0.6
    source = "tfidf"

    # -------- Priority fallback --------
    if priority_conf < THRESHOLD:
        priority, priority_conf = embedding_predict(
            text, PRIORITY_EMBEDDINGS
        )
        source = "embeddings"

    # -------- Privacy fallback --------
    if privacy_conf < 0.7:
        privacy, privacy_conf = embedding_predict(
            text, PRIVACY_EMBEDDINGS
        )
        source = "embeddings"
    return {
        "priority": priority,
        "priority_confidence": round(priority_conf, 3),
        "privacy": privacy,
        "privacy_confidence": round(privacy_conf, 3),
        "source": source
    }






# for run api use the command:
#cd d:/private/AI
#C:/Users/a.algazar/AppData/Local/Programs/Python/Python314/python.exe -m uvicorn api_classify_mail:app --reload
