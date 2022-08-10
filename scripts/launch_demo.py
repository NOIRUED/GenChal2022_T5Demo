import streamlit as st
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration, AutoModelForCausalLM, T5Tokenizer, AutoModelForMaskedLM
import torch

st.title("Demo of python code")
T5_PATH = "TMUUED/t5_fcg_2022"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t5_tokenizer = AutoTokenizer.from_pretrained(T5_PATH, use_fast=False)
t5_config = AutoConfig.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)

text = st.text_area('Text to Analyze', '''Input a text''')
start_analyse = st.button("Analyse")

if start_analyse:
    input_ids = t5_tokenizer(text, add_special_tokens=True, return_tensors="pt").input_ids.to(DEVICE)
    outputs = t5_mlm.generate(input_ids=input_ids, max_length=1024)
    _txt = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.text_area("", _txt)