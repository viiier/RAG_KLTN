from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

raw_text ="""Ngày 22/12, đại diện BCH Công đoàn cơ sở SHB đã đến thăm hỏi, động viên và trao quà cho gia đình chị Phạm Thị Mai - SHB TTKD - CBNV có hoàn cảnh đặc biệt. Đây cũng là một trong những truyền thống tốt đẹp của người SHB, luôn sẵn sàng giúp đỡ đồng nghiệp không may gặp khó khăn trong cuộc sống.
Tháng 5/2023 vừa qua, cháu Nguyễn Bảo Nguyên - con trai chị Phạm Thị Mai, Kiểm soát viên tại SHB TTKD, trên đường đi học về không may bị thanh sắt từ công trường đang thi công rơi xuống đầu gây chấn thương sọ não và giám định thương tật là 49% (Theo đánh giá của viện khoa học hình sự). Trải qua 02 ca phẫu thuật để ghép xương sọ nhân tạo, sức khỏe cháu Nguyên vẫn chưa ổn định, ảnh hưởng đến tình hình học tập và vấn đề tự sinh hoạt cá nhân. Chị Mai cũng là mẹ đơn thân và trụ cột kinh tế chính trong gia đình nên cuộc sống rất khó khăn và vất vả.


BCH Công Đoàn cơ sở SHB trực tiếp đến thăm hỏi và tặng quà cho gia đình

Nắm được thông tin hoàn cảnh ấy, Công đoàn cơ sở SHB đã xin ý kiến chỉ đạo từ Ban lãnh đạo, trực tiếp tới thăm hỏi và trao quà 90 triệu đồng cho gia đình chị Mai (Hỗ trợ từ Ban lãnh đạo và Quỹ Chia sẻ yêu thương). Số tiền này để phần nào hỗ trợ chi phí phẫu thuật, điều trị cho cháu Nguyên và cuộc sống sau này. Trước đó, Ban Giám đốc TTKD và CBNV TTKD cũng đã ủng hộ và giúp đỡ gia đình chị Mai 56 triệu đồng.

Trước sự quan tâm, chăm sóc đặc biệt của Ban lãnh đạo Ngân hàng cũng như Ban Giám đốc đơn vị, chị Mai vô cùng xúc động: “Trong thời gian khó khăn nhất, chính Ban lãnh đạo và đồng nghiệp tại SHB là những người luôn hỗ trợ công việc, động viên, khích lệ tinh thần để tôi và gia đình có thể yên tâm chăm sóc cho cháu. Tôi cảm thấy vô cùng biết ơn Ban lãnh đạo và BCH công đoàn đã luôn lắng nghe và quan tâm kịp thời đến đời sống của CBNV.”

“Truyền thống nhân văn cao đẹp của người SHB đã luôn được nuôi dưỡng và gìn giữ suốt hành trình 30 năm phát triển. Quỹ “Chia sẻ yêu thương SHB” - Quỹ vận động CBNV SHB toàn hệ thống đóng góp 1 ngày lương cơ bản/năm để hỗ trợ các HCKK của CBNV, người thân SHB đã được thành lập 5 năm và đã giúp đỡ rất nhiều hoàn cảnh khó khăn trên toàn hệ thống. Ban lãnh đạo cũng như BCH Công đoàn cơ sở rất mong muốn có thể kịp thời hỗ trợ CBNV, phần nào giúp đỡ anh chị em trong Đại gia đình SHB vượt qua nghịch cảnh cuộc sống và an tâm công tác.” - Chủ tịch Công đoàn cơ sở Phạm Thị Quỳnh Hoa chia sẻ.

Với tinh thần tương thân, tương ái, lá lành đùm lá rách, người SHB luôn đề cao chữ “Tâm” và giá trị nhân văn làm kim chỉ nam cho mọi hành động. Tập thể CBNV ngân hàng SHB thường xuyên chung tay giúp đỡ đồng nghiệp không may gặp khó khăn, đồng thời “Chia sẻ yêu thương” đến với những mảnh đời kém bất hạnh ngoài cộng đồng, xã hội. """
# get the text chunks
text_chunks = get_text_chunks(raw_text)
embeddings = GPT4AllEmbeddings()
# create vector store
#vectorstore = get_vectorstore(text_chunks)
vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
db = vectorstore.as_retriever(search_kwargs={'k': 3})

tokenizer = AutoTokenizer.from_pretrained("vinai/PhoGPT-4B-Chat")
model = AutoModelForCausalLM.from_pretrained("vinai/PhoGPT-4B-Chat")

generation_params = {
    "do_sample": True,
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 50,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1}

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_params
)

my_pipeline = HuggingFacePipeline(pipeline=pipe)
embeddings = GPT4AllEmbeddings()


template_qah = "Dựa vào ngữ cảnh sau để trả lời câu hỏi\n{context}\nvà lịch sử\n{chat_history}\n### Câu hỏi:\n{question}\n\n### Trả lời:"
prompt_qah = PromptTemplate(template=template_qah, input_variables=["question"])


template_qah_1 = "Lịch sử:\n{chat_history}\n### Câu hỏi:\n{question}\n\n### Trả lời:"
prompt_qah_1 = PromptTemplate(template=template_qah_1, input_variables=["question"])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Retrivial QA

qah_chain = ConversationalRetrievalChain.from_llm(llm=my_pipeline,
                                      retriever=db,
                                       return_source_documents=False,verbose=True,
                                 memory = memory,combine_docs_chain_kwargs={'prompt': prompt_qah},
     condense_question_prompt=prompt_qah_1,
                                        )

query = "Công đoàn cơ sở SHB đã làm gì vào ngày 22/12?"
sol=qah_chain({"question": query})
print(sol)