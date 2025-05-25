import torch

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(
                logits<min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature>0.0:
            logits=logits/temperature
            probas=torch.softmax(logits, dim=-1)
            idx_next=torch.multinomial(probas,num_samples=1)
        else:
            idx_next=torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next==eos_id:
            break
        idx=torch.cat((idx,idx_next),dim=-1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# idx is a (batch_size, n_tokens) array of indices in the current context
def generate_text_sample(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        """
        crops current context if it exceeds the supported context size
        eg: if LLM supports only 5 tokens and the context size is 10
        only 5 tokens are used as context
        """
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        """
        Focuses only on the last time step so that (batch_size, n_tokens, vocab_size)
        becomes (batch_size, vocab_size)
        """
        logits = logits[:,-1,:]
        # probas has shape (batch_size, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        # idx_next has shape (batch_size, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        """
        Append sampled index tot he running sequence
        where idx has shape (batch, n_tokens+1)
        """
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx
