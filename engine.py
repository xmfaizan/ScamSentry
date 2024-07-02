import streamlit as st
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import pandas as pd

st.markdown("""
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<h1 class='centered-title'>ScamSentry: Review Authenticator</h1>", unsafe_allow_html=True)


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits


# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model_name = 'bert-base-uncased'
    num_classes = 2
    model = BERTClassifier(bert_model_name, num_classes).to(device)
    model.load_state_dict(torch.load('bert_classifier_model.pth', map_location=device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    return model, tokenizer, device


model, tokenizer, device = load_model_and_tokenizer()


# Prediction function
def prediction(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

    return "Human-Written Review" if preds.item() == 1 else "Computer-Generated Review"


#input
test_text = st.text_area("Paste the review here", height=200)

#buttons
if st.button("Check"):
    if test_text:
        with st.spinner('Analysing...'):
            predict = prediction(test_text, model, tokenizer, device)

        st.write(f"Predicted: {predict}")
#gifs
        if predict == "Computer-Generated Review":
            st.image("gif1.gif", caption="Computer-Generated Review")
        else:
            st.image("gif2.gif", caption="Human-Written Review")
    else:
        st.warning("Please enter some text to classify.")

st.markdown("<h3 class='centered-title'>Scam or Legit, Lets find OUT!!!</h3>", unsafe_allow_html=True)
#st.header("Scam or Legit, Lets find OUT!!!")
#st.title("ScamSentry: Review Authenticity Checker")