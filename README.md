📰 News20 NLP Pipeline



A structured Natural Language Processing (NLP) pipeline built on the 20 Newsgroups dataset for:



🔹 Part 1 — Classic Text Classification (BoW / TF-IDF)



🔹 Part 2 — SentenceTransformer Embedding Classification



🔹 Part 3 — KMeans Clustering + Topic Tree Generation



This project demonstrates both supervised classification and unsupervised topic modeling using modern NLP techniques.



📚 Dataset

20 Newsgroups



~18,000 documents



20 balanced categories



Multi-class classification problem



Categories include:



comp.\*



rec.\*



sci.\*



talk.\*



misc.\*



The dataset is automatically downloaded via sklearn.datasets.fetch\_20newsgroups.



🗂 Project Structure

news20-nlp-pipeline/

│

├── scripts/

│   ├── run\_part1.py

│   ├── run\_part2.py

│   └── run\_part3.py

│

├── src/

│   ├── cluster\_utils.py

│   ├── cluster\_viz.py

│   ├── data\_loader.py

│   ├── embedding\_cache.py

│   ├── llm\_packets.py

│   ├── metrics.py

│   ├── part1\_classic.py

│   ├── part2\_embeddings.py

│   └── utils.py

│

├── outputs/

│   ├── part1/

│   ├── part2/

│   ├── part3/

│   └── cache/

│

├── requirements.txt

└── README.md





All generated files are saved under the outputs/ directory.



⚙️ Setup

1️⃣ Create Virtual Environment

python -m venv .venv





Activate it:



Windows

.venv\\Scripts\\activate



Mac / Linux

source .venv/bin/activate



2️⃣ Install Dependencies

pip install -r requirements.txt





If you want UMAP visualizations in Part 3:



pip install umap-learn



🚀 How to Run

🔹 Part 1 — Classic Text Classification



Uses:



CountVectorizer (BoW)



TF-IDF



Logistic Regression / other ML models



Confusion matrix + evaluation metrics



Example Commands

Run with TF-IDF

python -m scripts.run\_part1 --vectorizer tfidf --save\_confusion\_png



Run with Bag-of-Words

python -m scripts.run\_part1 --vectorizer bow --save\_confusion\_png



Outputs

outputs/part1/

├── confusion\_matrix\_best\_\*.png

├── run\_metadata.json

└── top\_confusion\_pairs.json



🔹 Part 2 — SentenceTransformer Classification



Uses:



all-MiniLM-L6-v2 embeddings



Cached embeddings (for faster reruns)



ML classifier



Confusion matrix + metrics



Example Command

python -m scripts.run\_part2 --st\_model all-MiniLM-L6-v2





Optional:



--batch\_size 64

--normalize



Outputs

outputs/part2/

├── confusion\_matrix\_best\_\*.png

├── run\_metadata.json

└── top\_confusion\_pairs.json





Embeddings are cached in:



outputs/cache/





If cache exists, embeddings are not recomputed.



🔹 Part 3 — Clustering + Topic Tree



Uses:



SentenceTransformer embeddings



Elbow method (K = 2–9)



KMeans clustering



2-level hierarchical clustering



TF-IDF fallback topic labeling



Optional PCA / UMAP visualization



LLM labeling packet generation



Example Commands

Basic run

python -m scripts.run\_part3



With embedding normalization

python -m scripts.run\_part3 --normalize



Force specific number of clusters

python -m scripts.run\_part3 --k\_override 6



Generate PCA cluster plots

python -m scripts.run\_part3 --plot pca



Generate UMAP plots (requires umap-learn)

python -m scripts.run\_part3 --plot umap



What Part 3 Does

Step A — Top-Level Clustering



Runs elbow search (K=2..9)



Selects optimal K



Clusters all documents



Step B — Second-Level Clustering



Identifies 2 largest clusters



Splits each into exactly 3 subclusters



Generates subtopic labels



Step C — Partial Topic Tree



Displays and saves simple hierarchical topic tree



Outputs

outputs/part3/

├── part3\_elbow.png

├── cluster\_scatter\_pca\_top.png

├── cluster\_scatter\_pca\_sub\_parentX.png

├── part3\_top\_clusters.json

├── part3\_subclusters.json

├── topic\_tree.txt

└── llm\_packets/



⏱ Runtime Notes



Part 1: Fast (<1 min)



Part 2: ~2–5 min (first run downloads model)



Part 3:



Embeddings: 2–4 min



Clustering: <1 min



UMAP: +1–2 min



Subsequent runs are faster due to embedding caching.



📜 License



MIT License — Educational Use.



You are free to use, modify, and extend this code for academic and learning purposes.

