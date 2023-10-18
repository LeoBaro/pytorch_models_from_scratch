import torch
from torch import nn
from math import sqrt


def scaled_dot_product_attention(q, k, v, mask=None):
    k = torch.transpose(k, 1, 2)
    q_k_scaled = torch.bmm(q, k) / sqrt(q.size(-1))
    att_w = nn.Softmax(dim=-1)(q_k_scaled)
    if mask is not None:
        att_w = att_w.masked_fill(mask == 0, float("-inf"))
    contextualized_emb = torch.bmm(att_w, v)
    return contextualized_emb

class AttentionHead(nn.Module):

    def __init__(self, emb_size, linear_out_features=None):
        super().__init__()
        if linear_out_features is None:
            linear_out_features = emb_size
        self.q_linear = nn.Linear(
            in_features=emb_size, out_features=linear_out_features
        )
        self.k_linear = nn.Linear(
            in_features=emb_size, out_features=linear_out_features
        )
        self.v_linear = nn.Linear(
            in_features=emb_size, out_features=linear_out_features
        )

    def forward(self, x, mask=None) -> torch.Tensor:
        q = self.q_linear(x)
        k = self.k_linear(x) 
        v = self.v_linear(x) 
        return scaled_dot_product_attention(q, k, v, mask=mask) 
    

        
class MultiAttentionHead(nn.Module):

    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.att_heads = nn.ModuleList([
            AttentionHead(emb_size, emb_size // num_heads) for i in range(num_heads)
        ])
        self.linear = nn.Linear(in_features=emb_size, out_features=emb_size)

    def forward(self, x, mask=None):
        x = torch.cat([att_head(x, mask) for att_head in self.att_heads], dim=-1)
        x = self.linear(x)
        return x
    


class Embeddings(nn.Module):

    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, emb_size)
        self.positional_embeddings = nn.Embedding(vocab_size, emb_size)
        self.layer_norm = nn.LayerNorm(emb_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        input_ids = input_ids.long()
        seq_lenght = input_ids.size(1)
        pos_ids = torch.arange(0, seq_lenght, dtype=torch.long).unsqueeze(0) # -> [[0,1,2,..,seq_lenght-1]]     
        token_embeddings = self.token_embeddings(input_ids)
        positional_embeddings = self.positional_embeddings(pos_ids)
        embeddings = self.layer_norm(token_embeddings+positional_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, emb_size, drop_out_prob):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=emb_size, out_features=4*emb_size)
        self.linear_2 = nn.Linear(in_features=4*emb_size, out_features=emb_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_out_prob) 

    def forward(self, x):
        x = self.linear_1(x)      
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.multi_attention_head = MultiAttentionHead(emb_size, num_heads)
        self.pos_feed_forward = PositionWiseFeedForwardLayer(emb_size, 0.5)
        self.layer_norm_1 = nn.LayerNorm(emb_size)
        self.layer_norm_2 = nn.LayerNorm(emb_size)
    
    def forward(self, x):
        x_tmp = self.layer_norm_1(x)
        x_tmp = self.multi_attention_head(x_tmp)
        x = x_tmp + x
        x_tmp = self.layer_norm_2(x)
        x_tmp = self.pos_feed_forward(x_tmp)
        x = x_tmp + x
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, emb_size, num_encoder_layers, num_heads_attention):
        super().__init__()
        self.input_embedding = Embeddings(vocab_size, emb_size)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(emb_size, num_heads_attention) for _ in range(num_encoder_layers)
        ])
    
    def forward(self, x):
        x = self.input_embedding(x)
        return [encoder(x) for encoder in self.encoder_layers]
                




class CrossAttentionHead(AttentionHead):

    def __init__(self, emb_size, linear_out_features=None):
        super().__init__(emb_size, linear_out_features)

    def forward(self, q, k, v, mask=None):
        q = self.q_linear(q)
        k = self.k_linear(k) 
        v = self.v_linear(v) 
        return scaled_dot_product_attention(q, k, v, mask=mask) 

class MultiCrossAttentionHead(MultiAttentionHead):

    def __init__(self, emb_size, num_heads):
        super().__init__(emb_size, num_heads)
        self.att_heads = nn.ModuleList([
            CrossAttentionHead(emb_size, emb_size // num_heads) for i in range(num_heads)
        ])
        self.linear = nn.Linear(in_features=emb_size, out_features=emb_size)

    def forward(self, q, k, v, mask=None):
        x = torch.cat([att_head(q, k, v, mask) for att_head in self.att_heads], dim=-1)
        x = self.linear(x)
        return x    


class TransformerDecoderLayer(nn.Module):
    
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.masked_multi_attention_head = MultiAttentionHead(emb_size, num_heads)
        self.multi_cross_attention_head = MultiCrossAttentionHead(emb_size, num_heads)
        self.pos_feed_forward = PositionWiseFeedForwardLayer(emb_size, 0.5)
        self.layer_norm_1 = nn.LayerNorm(emb_size)
        self.layer_norm_2 = nn.LayerNorm(emb_size)
        self.layer_norm_3 = nn.LayerNorm(emb_size)

    def forward(self, x, k, v, mask=None):
        x = self.masked_multi_attention_head(x, mask) + x
        x = self.layer_norm_1(x)
        x = self.multi_cross_attention_head(x, k, v) + x 
        x = self.layer_norm_2(x)
        x = self.pos_feed_forward(x) + x
        x = self.layer_norm_3(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size, emb_size, num_decoder_layers, num_heads_attention):
        super().__init__()
        self.input_embedding = Embeddings(vocab_size, emb_size)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(emb_size, num_heads_attention) for _ in range(num_decoder_layers)
        ])
    
    def forward(self, x, encoders_output):
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        x = self.input_embedding(x)
        for i, decoder in enumerate(self.decoder_layers):
            x = decoder(x, encoders_output[i], encoders_output[i], mask)
        return x
    

class Transformer(nn.Module):
    
    def __init__(self, vocab_size, emb_size, num_encoder_layers, num_decoder_layers, num_heads_attention): 
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, emb_size, num_encoder_layers, num_heads_attention)
        self.decoder = TransformerDecoder(vocab_size, emb_size, num_decoder_layers, num_heads_attention)

    def forward(self, x):
        encoders_output = self.encoder(x)
        return self.decoder(x, encoders_output)


class TransformerForClassification(nn.Module):

    def __init__(self, number_of_labels, vocab_size, emb_size, num_encoder_layers, num_heads_attention): 
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, emb_size, num_encoder_layers, num_heads_attention)
        self.classification_layer = nn.Linear(emb_size, number_of_labels)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.encoder(x)[-1][:, 0, :] # select only one hidden state (the cLS token one)
        x = self.dropout(x)
        x = self.classification_layer(x)
        return x    

if __name__ == "__main__":

 
    from transformers import DistilBertTokenizerFast, DistilBertModel
    from torchinfo import summary

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    tokens = tokenizer.encode(
        "Run, rabbit run. Dig that hole, forget the sun.", return_tensors="pt"
    ).long()

    #print(transf_enc(tokens).shape)


    #e = Embeddings(100000, 768)
    #summary(e, tokens.size(), device="cpu", verbose=1)

    #mah = MultiAttentionHead(emb.shape[-1], 3)
    #print("mah(emb).shape:", mah(emb).shape)

    #transf_enc = TransformerEncoder(100000, 768, 7, 3)
    #summary(transf_enc, tokens.size(), device="cpu", verbose=1)
 
    #t = Transformer(100000, 768, 6, 6, 8)
    #summary(t, tokens.size(), device="cpu", verbose=1)
    #print(t(tokens).shape)



    t = TransformerForClassification(5, 100000, 768, 6, 8)
    summary(t, tokens.size(), device="cpu", verbose=1)
    print(t(tokens).shape)
    print(t(tokens))


