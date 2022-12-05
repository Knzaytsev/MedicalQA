import torch

class Encoder(torch.nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, 
                 hidden_size, num_layers, dropout, padding_idx):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size,
                                             embedding_dim=embedding_dim,
                                             padding_idx=padding_idx)
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, 
                                  hidden_size=hidden_size, 
                                  num_layers=num_layers,
                                  dropout = dropout,
                                  batch_first=True, )
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(in_features=hidden_size, 
                                      out_features=hidden_size)
        
    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x, mem = self.lstm(x)
        x = self.linear(x)
        return x, mem
    
class Attention(torch.nn.Module):
    
    def __init__(self, hidden_size, max_len):
        super(Attention, self).__init__()
        
        self.attention_layer = torch.nn.Linear(in_features=hidden_size*2, out_features=max_len)
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, x, hidden):
        attention_weights = torch.cat((x, hidden[0].transpose(1, 0)), dim=-1)
        attention_weights = self.attention_layer(attention_weights)
        return self.softmax(attention_weights)
    
class Decoder(torch.nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, 
                 hidden_size, num_layers, dropout, 
                 padding_idx, attention_layer):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size,
                                             embedding_dim=embedding_dim,
                                             padding_idx=padding_idx)
        self.attention = attention_layer
        self.combine_layer = torch.nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size)
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, 
                                  hidden_size=hidden_size, 
                                  num_layers=num_layers, 
                                  dropout=dropout,
                                  batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(in_features=hidden_size, 
                                      out_features=vocab_size)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x, mem, encoder_outputs):
        emb = self.embeddings(x)
        emb = self.dropout(emb)
        weights = self.attention(emb, mem)
        x = torch.bmm(weights, encoder_outputs)
        x = torch.cat((emb, x), dim=-1)
        x = self.combine_layer(x)
        x = self.relu(x)
        x, mem = self.lstm(x, mem)
        x = self.linear(x)
        return x, mem, weights
    
class Seq2SeqModel(torch.nn.Module):
    
    def __init__(self, encoder, decoder, teacher_forcing=.5):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing = teacher_forcing
        
    def forward(self, x, y):
        encoder_outputs, encoder_hidden = self.encoder(x)
        
        decoder_hidden = encoder_hidden
        decoder_input = y[:, 0].unsqueeze(1)
        
        decoder_outputs = [torch.zeros((x.size(0), 1, decoder.vocab_size)).long().to(device)]
        for i in range(1, y.size(1)):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            #decoder_outputs.append(decoder_output)
            teacher_force = random.random() < self.teacher_forcing
            decoder_input = y[:, i].unsqueeze(1) if teacher_force else decoder_output.argmax(-1)
            decoder_outputs.append(decoder_output)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs