from aiogram import Bot, Dispatcher, executor, types
import youtokentome as yttm
import torch
import torch.nn.functional as F
import numpy as np
from chitchat_model import *

API_TOKEN = ''

TOKENIZER_MODEL_PATH = 'pretrained_bpe_lm_med.model'
MODEL_PATH = 'best_language_model_state_dict.pth'

MAX_LEN = 256
EMBEDDING_DIM_ENCODER = 256
EMBEDDING_DIM_DECODER = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT_ENCODER = .5
DROPOUT_DECODER = .5
PAD_INDEX = 0
BOS_INDEX = 2
EOS_INDEX = 3
MAX_SENT_LEN = 128
VOCAB_SIZE = 30_000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_tokenizer(model_path):
    tokenizer = yttm.BPE(model=model_path)
    return tokenizer

def load_model(model_path):
    attention = Attention(HIDDEN_SIZE, MAX_LEN).to(DEVICE)
    encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM_ENCODER, HIDDEN_SIZE, 
                      NUM_LAYERS, DROPOUT_ENCODER, PAD_INDEX).to(DEVICE)
    decoder = Decoder(VOCAB_SIZE, EMBEDDING_DIM_DECODER, HIDDEN_SIZE, 
                      NUM_LAYERS, DROPOUT_DECODER, PAD_INDEX, attention).to(DEVICE)
    model = Seq2SeqModel(encoder, decoder)
    model.load_state_dict(torch.load('best_language_model_state_dict.pth'))
    return model

def generate_answer(question, decoding_alg):
    model.eval()
    question = torch.tensor(tokenizer.encode([question])).long().to(DEVICE)

    with torch.no_grad():
        answer = tokenizer.decode(decoding_alg(question), ignore_ids=[PAD_INDEX, BOS_INDEX, EOS_INDEX])[0]
    return answer
    
def GreedyDecoding(question):
    encoder_outputs, encoder_hidden = model.encoder(question)
    encoder_outputs = torch.cat(
        (encoder_outputs, 
         torch.zeros((1, MAX_LEN - encoder_outputs.size(1), HIDDEN_SIZE)).to(DEVICE)
        ), dim=1)

    decoder_hidden = encoder_hidden

    decoder_input = torch.tensor([[BOS_INDEX]]).long().to(DEVICE)

    decoder_output, decoder_hidden, decoder_attention = model.decoder(decoder_input, decoder_hidden, encoder_outputs)

    answer = []
    for _ in range(0, MAX_SENT_LEN):
        decoder_output, decoder_hidden, decoder_attention = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_input = decoder_output.argmax(-1)
        if decoder_input.item() == 3:
            break
        answer.append(decoder_input.item())
    return answer


def BeamSearch(question):
    class BeamSearchNode():
        def __init__(self, prev_node, prob, input_token, hidden):
            self.prev_node = prev_node
            self.prob = prob
            self.input_token = input_token
            self.hidden = hidden
            self.depth = prev_node.depth + 1 if prev_node else 0
            self.prod_probs = prev_node.prod_probs + prob if prev_node else 0

        def __repr__(self):
            prob_str = 'prob: ' + str(np.round(self.prob, 2))
            input_token_str = 'token: ' + str(self.input_token)
            depth_str = 'depth: ' + str(self.depth)
            prod_probs_str = 'product probs: ' + str(np.exp(self.prod_probs))
            return '\n'.join([prob_str, input_token_str, depth_str, prod_probs_str])

        def build_sentence(self, sentence):
            sentence.append(self.input_token)
            if self.prev_node:
                self.prev_node.build_sentence(sentence)
    WIDTH = 10
    
    encoder_outputs, encoder_hidden = model.encoder(question)
    encoder_outputs = torch.cat(
                                (encoder_outputs, 
                                 torch.zeros((1, MAX_LEN - encoder_outputs.size(1), HIDDEN_SIZE)).to(DEVICE)
                                ), dim=1)

    decoder_hidden = encoder_hidden
    
    nodes = [BeamSearchNode(None, 0, BOS_INDEX, decoder_hidden)]
    for _ in range(MAX_SENT_LEN):
        cur_nodes = []
        cur_vals = []
        
        for node in nodes:
            input_token = torch.tensor([[node.input_token]]).long().to(DEVICE)
            decoder_output, decoder_hidden, attention_weights = model.decoder(input_token, node.hidden, encoder_outputs)
            top_decoder_vals, top_decoder_idxs = F.log_softmax(decoder_output, -1).topk(WIDTH, -1)
            top_decoder_vals, top_decoder_idxs = top_decoder_vals.detach().cpu(), top_decoder_idxs.detach().cpu()
            for idx in range(top_decoder_idxs.size(-1)):
                decoder_idx_val = top_decoder_vals[:, :, idx].item()
                decoder_idx_token = top_decoder_idxs[:, :, idx].item()
                new_node = BeamSearchNode(node, decoder_idx_val, decoder_idx_token, decoder_hidden)
                cur_nodes.append(new_node)
                cur_vals.append(decoder_idx_val)
        top_nodes = np.argpartition(cur_vals, -WIDTH)[-WIDTH:]
        nodes = np.array(cur_nodes)[top_nodes]
        eos_nodes = [i for i, node in enumerate(nodes) if node.input_token == 3]
        if len(eos_nodes) == WIDTH:
            break
        nodes = np.delete(nodes, eos_nodes)
        
    probs_per_node = [node.prod_probs for node in nodes]
    most_prob_node = nodes[np.argmax(probs_per_node)]
    sentence = []
    most_prob_node.build_sentence(sentence)
    return sentence[::-1]

tokenizer = load_tokenizer(TOKENIZER_MODEL_PATH)
model = load_model(MODEL_PATH)
decoding_method = GreedyDecoding

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    text = ["Привет!",
            "Я бот, отвечающий на вопросы, связанные с жалобами на здоровье!",
            "Отправь мне вопрос по жалобам на здоровье, а я тебе обязательно отвечу.",]
    await message.reply("\n".join(text))
    
@dp.message_handler(commands=['help'])
async def send_help(message: types.Message):
    text = ["Для получения ответа от бота введите любой текст и отправьте его.",
            "Чтобы выбрать способ генерации текста, введите перед генерацией один из способов:",
            "beamsearch",
            "greedy",
            "По умолчанию используется greedy decoding."]
    await message.reply("\n".join(text))
    
@dp.message_handler()
async def answer(message: types.Message):
    answer_message = ""
    global decoding_method
    if message['text'] == "beamsearch":
        decoding_method = BeamSearch
        answer_message = "Выбран алгоритм Beam Search"
    elif message['text'] == "greedy":
        decoding_method = GreedyDecoding
        answer_message = "Выбран алгоритм Greedy Decoding"
    else:
        answer_message = generate_answer(message['text'], decoding_method)
        
    await message.answer(answer_message)
    
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)