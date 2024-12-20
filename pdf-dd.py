from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
from rag_test import RAGSystem

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 200


@app.route('/RAG', methods=['GET', 'POST'])
def rag_chat():
    if request.method == 'GET':
        # チャットページのテンプレートを表示
        return render_template('chat.html')

    if request.method == 'POST':
        data = request.json
        if not data or 'file_path' not in data or 'query' not in data:
            return jsonify({"error": "file_path and query must be provided"}), 400

        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            return jsonify({"error": "OpenAI API key not found"}), 500

        pdf_path = data['file_path']
        query = data['query']

        try:
            # RAGシステムのセットアップ
            rag_system = RAGSystem(pdf_path=pdf_path, openai_api_key=openai_api_key)

            # ドキュメントとベクトルストアの準備
            documents = rag_system.load_documents()
            split_docs = rag_system.split_documents(documents)
            rag_system.create_vectorstore(split_docs)

            # 質問応答チェーンのセットアップ
            rag_system.setup_qa_chain()

            # 質問に回答
            answer, source_docs = rag_system.answer_query(query)

            # レスポンスを構築
            source_doc_summaries = [
                {"content": doc.page_content[:200], "metadata": doc.metadata}
                for doc in source_docs
            ]
            return jsonify({
                "answer": answer,
                "source_docs": source_doc_summaries
            }), 200

        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)

