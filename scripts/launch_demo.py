import streamlit as st
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration, AutoModelForCausalLM, T5Tokenizer, AutoModelForMaskedLM
import torch
import re
from word_forms.word_forms import get_word_forms
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

def pos_replace(tokens):
    for i in range(len(tokens)):
        tokens[i] = list(tokens[i])
        if tokens[i][0] == ';':
            tokens[i][1] = ';'
        elif tokens[i][0] == '--':
            tokens[i][1] = '--'
        elif tokens[i][0] == '-':
            tokens[i][1] = '-'
        elif tokens[i][0] == '...':
            tokens[i][1] = '...'
        elif tokens[i][0] == "!!":
            tokens[i][1] = "!!"
        elif tokens[i][0] == "???":
            tokens[i][1] = "???"
        tokens[i] = tuple(tokens[i])
    return tokens

def preprocess(text):
    text_pos = text.replace("[BOE]", "")
    text_pos = text_pos.replace("[EOE]", "")
    tokens = text_pos.split(" ")
    tokens = nltk.pos_tag(tokens)
    tokens = pos_replace(tokens)
    text = text.split(" ")
    pos_list = []
    for i in range(len(text)):
        rtext = tokens[i][0]
        pos_token = tokens[i][1]
        if text[i][len(text[i])-5:len(text[i])] == '[EOE]':
            rtext = rtext + "[EOE]"
            pos_token = pos_token + "[EOE]"
        if text[i][0:5] == '[BOE]':
            rtext = "[BOE]" + rtext
            pos_token = "[BOE]" + pos_token
        text[i] = rtext
        pos_list.append(pos_token)
    text = " ".join(text)
    text_pos = " ".join(pos_list)
    text = text + " POS_information: " + text_pos
    return text

def postprocess(text, pre_text):
    text = text.replace('*', '<')
    pre_text = pre_text.replace("[BOE]", "")
    pre_text = pre_text.replace("[EOE]", "")
    pre_text = pre_text.split(" ")
    gram_tokens = re.findall(r'<<(.*?)>>', text)

    for tok in gram_tokens:
        tok_s = tok.split(" ")
        if len(tok_s) >= 2 or tok in pre_text:
            continue
        else:
            tok = tok.lower()
            if tok in pre_text:
                continue 
            word_form = get_word_forms(tok)
            word_forms = list(word_form['n']) + list(word_form['a']) + list(word_form['v']) + list(word_form['r'])
            tok_bool = False
            for word in word_forms:
                if word in pre_text:
                    tok_bool = True
            if tok_bool:
                continue
            if tok.capitalize() in pre_text:
                pred = pred.replace("<<"+tok+">>", "<<"+tok.capitalize()+">>")
                continue
            if tok.lower() in pre_text:
                continue
            text = "<NO_COMMENT>"
            break
    return text


st.title("Demo of the system proposed in GenChal2022")
T5_PATH = "TMUUED/t5_fcg_2022"
DEVICE = torch.device('cpu')

@st.cache(allow_output_mutation=True)
def load_model():
    t5_tokenizer = AutoTokenizer.from_pretrained(T5_PATH, use_fast=False)
    t5_config = AutoConfig.from_pretrained(T5_PATH)
    t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)
    return t5_tokenizer, t5_mlm

pre_text = st.text_area('Text to Analyze', '''Input a text''')
start_analyse = st.button("Analyze")

if start_analyse:
    t5_tokenizer, t5_mlm = load_model()
    pre_text = preprocess(pre_text)
    text = "Generate a feedback comment: {}".format(pre_text) 
    input_ids = t5_tokenizer(text, add_special_tokens=True, return_tensors="pt").input_ids.to(DEVICE)
    outputs = t5_mlm.generate(input_ids=input_ids, max_length=1024)
    _txt = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    _txt = postprocess(_txt, pre_text)
    st.text_area("Output", _txt)
