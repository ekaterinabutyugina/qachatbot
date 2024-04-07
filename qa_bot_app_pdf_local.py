import streamlit as st
import openai
import yaml
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
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
st.title("📝 File Q&A with ChatGPT")

# Upload the file:
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md", "pdf"))

# Text input:
question = st.text_input(
    "Ask something about the article",
    value = "Can you give me a short summary?",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

article = ""

if uploaded_file and question and not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")

if uploaded_file and question and openai_api_key:
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
      # st.write(article)

    else:
      article = uploaded_file.read().decode()
      # st.write(article)

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
# ChatGPT Connection with increased answer length:
    openai.api_key = openai_api_key
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=my_prompt,
        max_tokens=200,
    )

    st.write("### Answer")
    st.write(response['choices'][0]['text'])

from time import time

if option ==  "Mistral":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    import torch
    
    
    st.write("### TBA")
    model = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model)
    mistral_generate = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    tic = time()
    sequences = mistral_generate(
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
    st.write(runtime)

if option ==  "Gemma":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    import torch

    # model_id = "google/gemma-2b"# 
    model_id = "google/gemma-2b-it"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=dtype,
    )

    chat = [
        { "role": "user", "content":  my_prompt },
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids = inputs.to(model.device),  max_new_tokens=150)

    st.write(
        tokenizer.decode(outputs[0])
        [len(chat[0]['content']) + 59: -5]
        )
        
   