from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import pipeline
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)

from langchain_community.llms import HuggingFacePipeline

generation_params = {
    "do_sample": True,
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 40,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1}


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_params
)

my_pipeline = HuggingFacePipeline(pipeline=pipe)

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
embeddings = GPT4AllEmbeddings()


from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
pdf_data_path = "./data" #sửa lại đường dẫn
loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(is_separator_regex="\n", chunk_size=200, chunk_overlap=0)
# kiểm tra encoding cho tiếng Việt, dùng thử văn bản tiếng Anh
chunks = text_splitter.split_documents(documents)

db = FAISS.from_documents(chunks, embeddings)
db = db.as_retriever(search_kwargs={'k': 3})

from langchain.prompts import PromptTemplate
template = '''<s>[INST] Dựa vào ngữ cảnh sau để trả lời câu hỏi liên quan đến Đại học Duy Tân. Chỉ trả lời bằng tiếng Việt và sử dụng thông tin được cung cấp, đừng trả lời thêm. Nếu không biết thì hãy trả lời rằng bạn không biết về thông tin đó{context}
{question} [/INST]
'''
prompt_qa1 = PromptTemplate(template = template, input_variables=["question"])

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_llm(llm=my_pipeline,
                                      retriever=db,
                                       return_source_documents=True,verbose=True,
                                prompt = prompt_qa1
                                        )

query = "e muốn hỏi về học phí ngành công nghệ phần mềm"
sol=qa_chain.invoke({"query": query})
print(sol)