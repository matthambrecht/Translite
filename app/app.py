from __future__ import unicode_literals, print_function, division
from typing import Union
from sklearn.pipeline import Pipeline
from flask import Flask, redirect, render_template, request, jsonify
from io import open
from typing import Union
from flask_cors import CORS

import os
import pickle
import json
import unicodedata
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

app = Flask(
    __name__,
    template_folder="../public",
    static_folder="../static")
app.logger.setLevel(logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORS(app)

# Proccessor Stuff


class Lang:
    def __init__(self, name: str):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence: str):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def indexesFromSentence(
        lang: Lang,
        sentence: str) -> list[int]:
    app.logger.error(sentence)
    app.logger.error([lang.word2index.get(word)
                     for word in sentence.split(' ')])
    return [lang.word2index.get(word) for word in sentence.split(
        ' ') if lang.word2index.get(word)]


def tensorFromSentence(
        lang: Lang,
        sentence: str) -> torch.tensor:
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def cleanStr(
        s: str,
        lang: Lang = None):  # No crash on my watch
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s.lower().strip())
        if unicodedata.category(c) != 'Mn'
    )
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)

    if lang:
        s = s.split()

        for word in s:
            if word not in lang.word2index:
                word = ""

        s = ' '.join(s)

    return s.strip()
# End Processor Stuff

# Begin Translator Models


class Encoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout_p: float = 0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 output_size: int):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size,
            1,
            dtype=torch.long,
            device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(20):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttentionDecoder(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 output_size: int,
                 dropout_p: float = 0.1):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size,
            1,
            dtype=torch.long,
            device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self,
                     input,
                     hidden,
                     encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
# End Translator Models

# Begin Pipeline Stuff


class Translite():
    def __init__(self,
                 classes: dict[int, str],
                 config: dict[str, str],
                 classifier: Pipeline,
                 models: dict[str, tuple]):
        self.config: dict[str, str] = config
        self.models: dict[str, tuple] = models
        self.classifier: Pipeline = classifier
        self.classes: dict[int, str] = classes

        global EOS_token, SOS_token, MAX_LENGTH
        EOS_token = int(config["EOS_token"])
        SOS_token = int(config["SOS_token"])
        MAX_LENGTH = int(config["max_sent_length"])

    def __classify(self,
                   text: str) -> str:
        return self.classes[
            f"{int(self.classifier.predict([text]).item())}"
        ]

    def __translate(
            self,
            input_lang: Lang,
            output_lang: Lang,
            encoder: Encoder,
            decoder: AttentionDecoder,
            text: str) -> str:
        with torch.no_grad():
            app.logger.debug(f"Model Input (ASCII) -> {text}")
            input_tensor = tensorFromSentence(input_lang, text)
            app.logger.debug(f"Model Input (Tokenized) -> {input_tensor}")

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = decoder(
                encoder_outputs, encoder_hidden)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            decoded_words = []
            for idx in decoded_ids:
                if idx.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break

                decoded_words.append(output_lang.index2word[idx.item()])

            app.logger.debug(f"Model Output -> {decoded_words}")

            # I don't like how hacky this is but meh
            output_sentence = ' '.join(
                decoded_words[:min(len(decoded_words), len(text.split(' ')))])

            return output_sentence

    def run(self,
            text: str) -> Union[str, str]:
        class_name = self.__classify(text)
        app.logger.debug(f"Classifying \"{text}\" as \'{class_name}\'")
        input_lang, output_lang, encoder, decoder = self.models[class_name]
        text = cleanStr(text, lang=input_lang)

        if not any([input_lang.word2index.get(word.strip())
                   for word in text.split(' ')]):
            return class_name, "Text too short to translate"

        if len(text.split(' ')) > MAX_LENGTH:
            return class_name, "Text too long to translate"

        text = [word.strip() for word in text.split(
            ' ') if input_lang.word2index.get(word.strip())]

        return class_name, self.__translate(
            input_lang=input_lang,
            output_lang=output_lang,
            encoder=encoder,
            decoder=decoder,
            text=' '.join(text)
        )


def loadTranslite(folder: str = '') -> Translite:
    translation_models = {}

    folder = f"{os.getcwd()}/{folder}"
    app.logger.debug(f"Checking Directory: {folder}")

    if not os.path.isfile(f"{folder}/config.json"):
        raise FileNotFoundError("config.json not found")

    if not os.path.isfile(f"{folder}/classes.json"):
        raise FileNotFoundError("classes.json not found")

    if not os.path.isfile(f"{folder}/classifier.pkl"):
        raise FileNotFoundError("classifier.pkl not found")

    with open(f"{folder}/config.json", "r") as f:
        config = json.load(f)

    with open(f"{folder}/classes.json", "r") as f:
        classes = json.load(f)

    with open(f"{folder}/classifier.pkl", "rb") as f:
        classifier_model = pickle.load(f)

    for class_name in classes.values():
        if not os.path.isfile(f"{folder}/{class_name}_encoder.pt"):
            raise FileNotFoundError(f"{class_name}_encoder.pt not found")

        if not os.path.isfile(f"{folder}/{class_name}_decoder.pt"):
            raise FileNotFoundError(f"{class_name}_decoder.pt not found")

        if not os.path.isfile(f"{folder}/{class_name}_class.pkl"):
            raise FileNotFoundError(f"{class_name}_class.pkl not found")

        if not os.path.isfile(f"{folder}/{class_name}_class_out.pkl"):
            raise FileNotFoundError(f"{class_name}_class_out.pkl not found")

        input_lang = pickle.load(
            open(f"{folder}/{class_name}_class.pkl", "rb"))
        output_lang = pickle.load(
            open(f"{folder}/{class_name}_class_out.pkl", "rb"))
        encoder = Encoder(input_lang.n_words, config["hidden_size"]).to(device)
        decoder = AttentionDecoder(
            config["hidden_size"],
            output_lang.n_words).to(device)
        encoder.load_state_dict(
            torch.load(
                f"{folder}/{class_name}_encoder.pt",
                map_location=device))
        decoder.load_state_dict(
            torch.load(
                f"{folder}/{class_name}_decoder.pt",
                map_location=device))
        translation_models[class_name] = (
            input_lang, output_lang, encoder, decoder)

    return Translite(classes,
                     config,
                     classifier_model,
                     translation_models)
# End Pipeline Stuff


pipeline: Translite = loadTranslite("binaries")


@app.errorhandler(404)
def not_found(e):
    return redirect('/')


@app.route('/translate', methods=['POST'])
def translate():
    data = request.json  # Get JSON data from request
    text = data.get('text')
    classification, translated = pipeline.run(text=text)
    return jsonify({'language': classification, 'translated': translated})


@app.route("/")
def home():
    app.logger.debug("Rendering homepage")
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
