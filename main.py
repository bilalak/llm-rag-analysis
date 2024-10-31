from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from ai.chain import get_ai_response


app = FastAPI()

store = {}

@app.get("/")
async def get_status():
    return {"status": "running"}


# Chat
@app.post("/api/chat")
async def sse_request(request: Request):
    try:
        data = await request.json()
        message = data.get("message")
        session_id = data.get("session_id")
        response = get_ai_response(message, session_id, store)
        return JSONResponse(status_code=200, content={"content": response})
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})


# Chain Chat
@app.post("/api/chain-chat")
async def chain_chat(request: Request):
    try:
        data = await request.json()
        message = data.get("message")
        response = get_ai_response(message)
        return JSONResponse(status_code=200, content={"content": response})
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})


if __name__ == "__main__":
    app.run()
