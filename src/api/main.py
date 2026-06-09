from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Import từ các module bạn đã viết
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger
from core.llm.manager import LLMManager
from core.retriever.factory import RetrieverFactory
from core.reranker.factory import RerankerFactory
from models.query_parser import QueryParser
from models.retriever import DocumentRetriever
from models.reranker import DocumentReranker
from models.answer_generator import AnswerGenerator
from pipeline.rag_orchestrator import RAGOrchestrator

logger = get_logger("api.main")

app = FastAPI(title="AI Code Assistant API")

# Biến toàn cục lưu Orchestrator
pipeline = None

class ChatRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup_event():
    """Khởi tạo toàn bộ mô hình AI và kết nối DB khi Server bắt đầu chạy."""
    global pipeline
    logger.info("Initializing System Components...")
    
    llm_manager = LLMManager()
    core_retriever = RetrieverFactory.create_retriever(GLOBAL_CONFIG.get("retriever", {}))
    core_reranker = RerankerFactory.create_reranker(GLOBAL_CONFIG.get("models", {}).get("reranker", {}))

    query_parser = QueryParser(llm=llm_manager.parser_llm)
    answer_generator = AnswerGenerator(llm=llm_manager.generator_llm)
    retriever_agent = DocumentRetriever(retriever_model=core_retriever)
    reranker_agent = DocumentReranker(reranker_model=core_reranker)

    pipeline = RAGOrchestrator(
        query_parser=query_parser,
        retriever=retriever_agent,
        reranker=reranker_agent,
        answer_generator=answer_generator
    )
    logger.info("System Initialization Complete.")

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """Endpoint nhận câu hỏi và trả về câu trả lời từ RAG Pipeline."""
    try:
        answer = pipeline.run(request.query)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error during RAG pipeline: {e}")
        return {"answer": f"Xin lỗi, đã xảy ra lỗi hệ thống: {str(e)}"}

# @app.get("/", response_class=HTMLResponse)
# def get_web_ui():
#     """Trang Web Demo Mini tích hợp sẵn giao diện Chat."""
#     html_content = """
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>AI Code Assistant</title>
#         <style>
#             body { font-family: Arial, sans-serif; max-width: 800px; margin: auto; padding: 20px; background-color: #f4f4f9; }
#             #chat-box { height: 500px; overflow-y: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }
#             .msg { margin-bottom: 15px; line-height: 1.5; }
#             .user-msg { color: #0056b3; font-weight: bold; }
#             .ai-msg { color: #333; background: #e9ecef; padding: 10px; border-radius: 5px; }
#             .input-area { display: flex; gap: 10px; }
#             input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 16px; }
#             button { padding: 10px 20px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
#             button:hover { background: #218838; }
#         </style>
#     </head>
#     <body>
#         <h2>🤖 AI Code Assistant (RAG Pipeline)</h2>
#         <div id="chat-box"></div>
#         <div class="input-area">
#             <input type="text" id="query" placeholder="Hỏi về thư viện Python (VD: Làm sao đọc file csv trong pandas?)" onkeypress="if(event.key === 'Enter') sendMessage()">
#             <button onclick="sendMessage()">Gửi</button>
#         </div>

#         <script>
#             async function sendMessage() {
#                 const inputField = document.getElementById("query");
#                 const query = inputField.value.trim();
#                 if (!query) return;

#                 const chatBox = document.getElementById("chat-box");
#                 chatBox.innerHTML += `<div class="msg user-msg">Bạn: ${query}</div>`;
#                 inputField.value = "";
                
#                 // Hiển thị trạng thái đang gõ
#                 const typingId = "typing-" + Date.now();
#                 chatBox.innerHTML += `<div id="${typingId}" class="msg ai-msg">AI đang suy nghĩ (Retrieving & Generating)...</div>`;
#                 chatBox.scrollTop = chatBox.scrollHeight;

#                 try {
#                     const response = await fetch("/api/chat", {
#                         method: "POST",
#                         headers: { "Content-Type": "application/json" },
#                         body: JSON.stringify({ query: query })
#                     });
#                     const data = await response.json();
                    
#                     document.getElementById(typingId).remove();
#                     // Thay thế \n thành thẻ <br> để xuống dòng HTML
#                     const formattedAnswer = data.answer.replace(/\\n/g, "<br>");
#                     chatBox.innerHTML += `<div class="msg ai-msg">AI: <br>${formattedAnswer}</div>`;
#                 } catch (err) {
#                     document.getElementById(typingId).innerHTML = "Lỗi kết nối đến Server.";
#                 }
#                 chatBox.scrollTop = chatBox.scrollHeight;
#             }
#         </script>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content)
@app.get("/", response_class=HTMLResponse)
def get_web_ui():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Code Assistant</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: auto; padding: 20px; background: #f4f4f9; }
            #chat-box { height: 500px; overflow-y: auto; background: white; padding: 20px; border-radius: 8px; }
            .msg { margin-bottom: 10px; }
            .user-msg { color: blue; font-weight: bold; }
            .ai-msg { background: #eee; padding: 10px; border-radius: 5px; }
            .input-area { display: flex; gap: 10px; }
            input { flex: 1; padding: 10px; }
            button { padding: 10px 20px; cursor: pointer; }
        </style>
    </head>

    <body>
        <h2>🤖 AI Code Assistant (RAG)</h2>

        <div id="chat-box"></div>

        <div class="input-area">
            <input type="text" id="query" placeholder="Nhập câu hỏi..."
                onkeypress="if(event.key==='Enter'){ window.sendMessage(); }">

            <button onclick="window.sendMessage()">Gửi</button>
        </div>

        <script>
            // ⭐ FIX QUAN TRỌNG: gán vào window để luôn global
            window.sendMessage = async function () {
                console.log("sendMessage triggered");

                const input = document.getElementById("query");
                const query = input.value.trim();
                if (!query) return;

                const chatBox = document.getElementById("chat-box");

                chatBox.innerHTML += `<div class="msg user-msg">Bạn: ${query}</div>`;
                input.value = "";

                const typingId = "typing-" + Date.now();
                chatBox.innerHTML += `<div id="${typingId}" class="msg ai-msg">AI đang xử lý...</div>`;

                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const response = await fetch("/api/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ query: query })
                    });

                    if (!response.ok) {
                        throw new Error("HTTP error " + response.status);
                    }

                    const data = await response.json();

                    document.getElementById(typingId).remove();

                    const answer = data?.answer ?? "No response from server";
                    const formatted = answer.replace(/\\n/g, "<br>");

                    chatBox.innerHTML += `
                        <div class="msg ai-msg">AI:<br>${formatted}</div>
                    `;

                } catch (err) {
                    console.error(err);
                    document.getElementById(typingId).innerHTML =
                        "❌ Lỗi kết nối server";
                }

                chatBox.scrollTop = chatBox.scrollHeight;
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
if __name__ == "__main__":
    # Để chạy local không cần docker: python src/api/main.py
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)