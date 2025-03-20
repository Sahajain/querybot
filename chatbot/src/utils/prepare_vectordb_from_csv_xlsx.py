import os
import pandas as pd
from utils.load_config import LoadConfig
import tiktoken

class PrepareVectorDBFromTabularData:
    def __init__(self, file_directory: str) -> None:
        self.APPCFG = LoadConfig()
        self.file_directory = file_directory

    def run_pipeline(self):
        self.df, self.file_name = self._load_dataframe(self.file_directory)
        self.docs, self.metadatas, self.ids, self.embeddings = self._prepare_data_for_injection(self.df, self.file_name)
        self._inject_data_into_chromadb()
        self._validate_db()

    def _inject_data_into_chromadb(self):
        collection = self.APPCFG.chroma_client.create_collection(name=self.APPCFG.collection_name)
        collection.add(
            documents=self.docs,
            metadatas=self.metadatas,
            embeddings=self.embeddings,
            ids=self.ids
        )
        print("==============================")
        print("Data is stored in ChromaDB.")

    def _load_dataframe(self, file_directory: str):
        file_names_with_extensions = os.path.basename(file_directory)
        print(file_names_with_extensions)
        file_name, file_extension = os.path.splitext(file_names_with_extensions)

        if file_extension == ".csv":
            df = pd.read_csv(file_directory)
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_directory)
        else:
            raise ValueError("The selected file type is not supported")
        
        return df, file_name

    # 🔹 Fixed count_tokens() - Now it's a method with `self`
    def count_tokens(self, text):
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _prepare_data_for_injection(self, df: pd.DataFrame, file_name: str):
        docs, metadatas, ids, embeddings = [], [], [], []

        for index, row in df.iterrows():
            output_str = "\n".join([f"{col}: {row[col] if pd.notna(row[col]) else 'null'}" for col in df.columns])

            print(f"🔍 Debug - Sending to OpenAI: {output_str}")
            print(f"🔍 Data type: {type(output_str)}")

            token_count = self.count_tokens(output_str)
            print(f"🔍 Token Count: {token_count}")

            if token_count > 8192:
                print("⚠️ Input too long! Skipping this row.")
                continue

            # Debugging print
            print(f"🔍 Type of input being sent: {type(output_str)}")
            print(f"🔍 Value of input: {output_str}")

            response = self.APPCFG.azure_openai_client.embeddings.create(
                input=[str(output_str)],  # ✅ Ensure it's a string inside a list
                model=self.APPCFG.embedding_model_name
            )

            embeddings.append(response.data[0].embedding)
            docs.append(output_str)
            metadatas.append({"source": file_name})
            ids.append(f"id{index}")

        return docs, metadatas, ids, embeddings


    def _validate_db(self):
        vectordb = self.APPCFG.chroma_client.get_collection(name=self.APPCFG.collection_name)
        print("==============================")
        print("Number of vectors in vectordb:", vectordb.count())
        print("==============================")
