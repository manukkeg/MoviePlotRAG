import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


class MovieRAGSystem:
    def __init__(self, hf_token: str = None):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        self.index = None
        self.chunks = []
        self.metadata = []
        
        self.hf_token = os.getenv('HF_TOKEN')
        
        if not self.hf_token:
            raise ValueError("HuggingFace token not found!")
                
        self.model_options = [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]
        
        self.client = InferenceClient(token=self.hf_token)
        self.model = self.model_options[0]
           
    def load_and_preprocess(self, csv_path: str, n_rows: int = 300) -> pd.DataFrame:      
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', nrows=n_rows)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1', nrows=n_rows)
        
        required_cols = ['Title', 'Plot']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Required columns {required_cols} not found!")
        
        df = df[required_cols].dropna()
        df = df[df['Plot'].str.strip() != '']
        return df
    
    def chunk_documents(self, docs: List[str], titles: List[str]) -> tuple:

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=lambda text: len(text.split()),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        all_chunks = []
        all_metadata = []
        
        for title, doc in zip(titles, docs):
            langchain_doc = Document(page_content=doc, metadata={"title": title})
            chunks = text_splitter.split_documents([langchain_doc])
            
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk.page_content)
                all_metadata.append({
                    'title': title,
                    'chunk_id': idx,
                    'text': chunk.page_content
                })
        return all_chunks, all_metadata
    
    def build_vector_store(self, df: pd.DataFrame):
        
        docs = df['Plot'].tolist()
        titles = df['Title'].tolist()
        
        self.chunks, self.metadata = self.chunk_documents(docs, titles)
        
        embeddings = self.embedding_model.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        print(f"Vector store build ")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                'text': self.chunks[idx],
                'title': self.metadata[idx]['title'],
                'distance': float(distance),
                'chunk_id': self.metadata[idx]['chunk_id']
            })
        
        return results
    
    def generate_answer(self, query: str, contexts: List[Dict]) -> Dict:
        context_str = "\n\n".join([
            f"Movie: {ctx['title']}\nPlot: {ctx['text']}"
            for ctx in contexts
        ])
        
        messages = [
            {
                "role": "system",
                "content": "You are a movie expert. Answer questions based ONLY on the provided movie plot contexts. Be concise and reference specific movie titles."
            },
            {
                "role": "user", 
                "content": f"""Movie Plot Contexts:
{context_str}

Question: {query}

Provide:
1. A clear answer (2-3 sentences) based on the contexts
2. Brief reasoning

Format:
ANSWER: [your answer]
REASONING: [your reasoning]"""
            }
        ]
        
        answer = ""
        reasoning = ""
        
        try:
            print("Calling LLM...")
            
            response = self.client.chat_completion(
                model=self.model,
                messages=messages,
                max_tokens=400,
                temperature=0.7,
            )
            
            full_response = response.choices[0].message.content.strip()
            
            if "ANSWER:" in full_response and "REASONING:" in full_response:
                parts = full_response.split("REASONING:")
                answer = parts[0].replace("ANSWER:", "").strip()
                reasoning = parts[1].strip() if len(parts) > 1 else ""
            else:
                answer = full_response
                movie_titles = list(set([ctx['title'] for ctx in contexts]))
                reasoning = f"Based on {len(contexts)} chunks from: {', '.join(movie_titles[:3])}"
                
        except Exception as e:
            error_msg = str(e)
            print(f"LLM Error: {error_msg[:150]}")
                    
        return {
            "answer": answer,
            "contexts": [f"{ctx['title']}: {ctx['text'][:150]}..." for ctx in contexts],
            "reasoning": reasoning
        }
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        contexts = self.retrieve(question, top_k)    
        result = self.generate_answer(question, contexts)
        return result


def main():
    if not os.getenv('HF_TOKEN'):
        print("HF_TOKEN not found!")
        return
    
    try:
        rag = MovieRAGSystem()
    except Exception as e:
        print(f"Init failed: {e}")
        return
    
    csv_path = "wiki_movie_plots_deduped.csv"
    
    if not os.path.exists(csv_path):
        print(f"Dataset not found: {csv_path}")
        return
    
    try:
        df = rag.load_and_preprocess(csv_path, n_rows=300)
        rag.build_vector_store(df)
    except Exception as e:
        print(f"Build failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("   Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            user_query = input("\n🎬 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_query:
            continue
        
        try:
            result = rag.query(user_query, top_k=3)
            print("\nRESULT:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()