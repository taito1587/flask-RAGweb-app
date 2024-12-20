import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.chat_models import ChatOpenAI

class RAGSystem:
    def __init__(self, pdf_path: str, openai_api_key: str, persist_directory: str = "chroma_db"):
        """
        RAGシステムの初期化
        :param pdf_path: 読み込むPDFファイルのパス
        :param openai_api_key: OpenAIのAPIキー
        :param persist_directory: Chromaベクトルストアの永続化ディレクトリ
        """
        self.pdf_path = pdf_path
        self.openai_api_key = openai_api_key
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vectorstore = None
        self.qa_chain = None

    def load_documents(self):
        """PDFファイルからドキュメントをロード"""
        loader = PyPDFLoader(self.pdf_path)
        return loader.load()
        
    def split_documents(self, documents, chunk_size=1000, chunk_overlap=100):
        """
        ドキュメントを指定されたチャンクサイズとオーバーラップ量で分割
        :param chunk_size: 
        テキストを処理する際の入力データを分割する一塊（チャンク）の大きさ。
        :param chunk_overlap:
        チャンク間で重複させる部分のサイズであり、情報のつながりや文脈を保持するために利用されます。
        オーバーラップが大きいと文脈をより多く維持できますが、処理コストが増加します。
        """
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    def create_vectorstore(self, documents):
        """ベクトルストアを作成"""
        self.vectorstore = Chroma.from_documents(documents, embedding=self.embeddings, persist_directory=self.persist_directory)

    def setup_qa_chain(self):
        """OpenAIのチャットボットをセットアップ"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore is not initialized. Call create_vectorstore() first.")
        
        # ベクトルストアのリトリーバーを取得
        retriever = self.vectorstore.as_retriever()

        # OpenAIのメモリをセットアップRARA
        memory = ConversationBufferMemory(
        input_key="question",
        memory_key="chat_history",  # 過去の会話履歴を保持
        return_messages=True,       # メッセージ履歴を返却
        output_key="answer"         # メモリに保存する出力キーを明示的に指定
    )

        # OpenAIのチャットモデルを使用した会話型質問応答チェーン
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                model="gpt-3.5-turbo",  # GPT-4 を利用（必要に応じて gpt-3.5-turbo に変更可）
                openai_api_key=self.openai_api_key
            ),
            retriever=retriever,
            memory=memory,
            return_source_documents=True  # ソースドキュメントも返却
        )

    def answer_query(self, query: str):
        """
        ユーザーの質問に回答
        :param query: ユーザーの質問
        :return: 回答と関連ソースドキュメント
        """
        if self.qa_chain is None:
            raise ValueError("QA Chain is not initialized. Call setup_qa_chain() first.")
    
    # OpenAIのチャットAPIを使用して応答を生成
        result = self.qa_chain.invoke({"question": query})
        return result["answer"], result["source_documents"]