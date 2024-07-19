import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import SystemMessage
from langchain.agents import create_openai_functions_agent, AgentExecutor
from utils import ChatbotConfig, ArgumentParser


def initialize_sales_bot(llm,vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history, enable_chat, llm):
    print(f"[message]{message}")
    print(f"[history]{history}")

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"]:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    elif enable_chat==True:
        system_message = SystemMessage(content="你是一个非常专业的服装导购，请用平易近人的态度，自然地回答客户关于服装的问题。你是一个人类，请不要透露出任何关于机器人/虚拟/智能这一类的信息。")
        answer = llm.invoke([system_message, message])
        print(answer)
        return answer.content
    else:
        return "不好意思，这个问题我暂时无法解答。"
    

def launch_gradio(enable_chat: bool, llm):
    demo = gr.ChatInterface(
        fn=lambda message, history : sales_chat(message, history, enable_chat, llm),
        title="服装销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()
    config = ChatbotConfig()
    config.initialize(args)
    enable_chat = args.enable_chat if args.enable_chat else config.enable_chat
    print(args, enable_chat)

    # 初始化服装销售机器人
    initialize_sales_bot(llm)
    # 启动 Gradio 服务
    launch_gradio(enable_chat, llm)
