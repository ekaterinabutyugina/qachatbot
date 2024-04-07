import streamlit as st
import openai
import yaml
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from functools import cache
from time import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# Open Credentials

try:
    with open('chatgpt_api_credentials.yml', 'r') as file:
        creds = yaml.safe_load(file)
except:
    creds = {}

# Open Sidebar
with st.sidebar:
    openai_api_key = creds.get('openai_key', '')

    if openai_api_key:
        st.text("OpenAI API Key provided")
    else:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    # adding a hyperlink
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


# Set Title:
st.title("📝 File Q&A with LLMs")

st.write("This app will help you to chat with your documents which are in `txt`, `md`, or `pdf` formats. Simply upload your file, choose a model and ask your question.")
st.info(" **Note**: \n - for OpeanAI model you'll need to provide your credentials. \n - for Mistral and Gemini expect to have a better performance with a local cuda-compatible GPU") 

# Upload the file:
uploaded_file = st.file_uploader(" **Upload an article** ", type=("txt", "md", "pdf"))

# Text input:
question = st.text_input(
    " **Ask something about the article. For example:**",
    value = "Can you give me a short summary?",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

article = ""

if uploaded_file and question:# and openai_api_key:
    if uploaded_file.name.split(".")[-1] == "pdf":
      rsrcmgr = PDFResourceManager()
      retstr = StringIO()
      laparams = LAParams()
      device = TextConverter(rsrcmgr, retstr, laparams=laparams)
      interpreter = PDFPageInterpreter(rsrcmgr, device)
      file_pages = PDFPage.get_pages(uploaded_file,check_extractable=False)
      for page in file_pages:
        interpreter.process_page(page)

      article = retstr.getvalue()
      device.close()
      retstr.close()

    else:
      article = uploaded_file.read().decode()

# Prompting
my_prompt = f"""Here's an article:{article}.\n\n
    \n\n\n\n{question}"""


option = st.selectbox(
        "Choose the model",
        ("OpenAI", "Mistral", "Gemma"),
      #  label_visibility=st.session_state.visibility,
       # disabled=st.session_state.disabled,
    )

if option == "OpenAI":

    if uploaded_file and question and not openai_api_key:
         st.info("Please add your OpenAI API key to continue.")

    if uploaded_file and question and openai_api_key:
         
    # ChatGPT Connection with increased answer length:
        tic = time()   
        openai.api_key = openai_api_key
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=my_prompt,
            max_tokens=200,
        )
        runtime = time() - tic
        st.write("### Answer from ChatGPT")
        st.write(response['choices'][0]['text'])
        st.write("Time for answer generation is ", "{:.2f}".format(runtime), "seconds")

@st.cache_resource 
def get_mistral_model_and_tokenizer(model_id): 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    return model, tokenizer

if option ==  "Mistral":

    has_cuda = torch.cuda.is_available()
    
    if not has_cuda:
        st.write("Please use another lighter model")

    else:
        st.write("### Answer from Mistral")
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"

        model, tokenizer =  get_mistral_model_and_tokenizer(model_id)

        tic = time()
        sequences = model(
        my_prompt,
        max_length=1024,
        truncation='only_first',
        do_sample=True,
        return_full_text=False,
        temperature=0.01
        )
        toc = time()
        runtime = toc-tic
        st.write(sequences[0]['generated_text'])
        st.write("Time for answer generation is ", "{:.2f}".format(runtime), "seconds")

# following line is to cache the model after it is loaded at least once
@st.cache_resource 
def get_gemma_model_and_tokenizer(model_id, has_cuda): 
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda" if has_cuda else "cpu",
        torch_dtype=dtype,
    )
    return model, tokenizer

if option ==  "Gemma":
    from huggingface_hub import login
    login(st.secrets["HF_token"])

    has_cuda = torch.cuda.is_available()

   
    st.write("### Answer from Gemma")
    model_id = "google/gemma-2b-it"

    model, tokenizer = get_gemma_model_and_tokenizer(model_id, has_cuda)

    chat = [
        { "role": "user", "content":  my_prompt },
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    
    tic = time()
    outputs = model.generate(input_ids = inputs.to(model.device),  max_new_tokens=150)
    runtime = time() - tic

    st.write(
        tokenizer.decode(outputs[0])
        [len(chat[0]['content']) + 59: -5]
        )
    st.write("Time for answer generation is ", "{:.2f}".format(runtime), "seconds")

        
   