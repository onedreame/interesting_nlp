import os
import sys
import torch
import random
import argparse
sys.path.append(os.getcwd())

import torch.nn.functional as F
from datautils import LSCCDataSet
from scripts.run_generation import GPT2Proto

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'),
                  filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now
    top_k = min(top_k, logits.size(-1))
    # print('initial: ', logits[:10])
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        # print(f"top k: {logits[:10]} sum:{(logits!=filter_value).sum()}")

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # print(f"sorted logits:{sorted_logits[:args.top_k+4]}")
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # print(cumulative_probabilities[:10])

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    # print(f"final: {logits[:10]} elements: {(logits!=filter_value).sum()}")
    # if temperature is not None:
    #     logits /= temperature
    return logits

def generate(args, history, tokenizer, model, speaker1, speaker2, special_token_ids):
    current_output = []
    with torch.no_grad():
        for i in range(args.max_len):
            input_dict = LSCCDataSet.encode(history, [current_output], tokenizer, speaker1, speaker2, do_encode=False)
            new_user_input_ids, token_type_ids = input_dict['input_ids'], input_dict['token_type_ids']
            new_user_input_ids_tensor = torch.tensor(new_user_input_ids, dtype=torch.long).unsqueeze(0).cuda()
            token_type_ids_tensor = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).cuda()
            outputs = model(new_user_input_ids_tensor, token_type_ids=token_type_ids_tensor)
            print(f"input:{tokenizer.convert_ids_to_tokens(new_user_input_ids)}")
            logits = outputs[0][0,-1,:] / args.temperature
            # logits = outputs[0][0, -1, :]
            logits = top_filtering(logits, args.top_k,top_p=args.top_p)
            probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, 1)
            # print(f"step:{i} token:{tokenizer.convert_ids_to_tokens(prev.item())} output[0]:{outputs[0].size()}"
            #       f"current output:{tokenizer.convert_ids_to_tokens(current_output)}")
            if i < args.min_length and prev.item() in special_token_ids:
                while prev.item() in special_token_ids:
                    print("resample: ", tokenizer.convert_ids_to_tokens(prev.item()))
                    # exit(0)
                    prev = torch.multinomial(probs, 1)
            if prev.item() in special_token_ids:
                break
            current_output.append(prev.item())
    return current_output

def interact(args, model, tokenizer):
    user_inputs = input(">> user:")
    speaker1, speaker2 = LSCCDataSet.get_identity_id(tokenizer)

    special_token_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, speaker2, speaker1]
    tokenizer.add_special_tokens({"additional_special_tokens": [tokenizer.convert_ids_to_tokens(_id)
                                                                for _id in [speaker1, speaker2]]})
    print(tokenizer.all_special_tokens)
    history = []
    while user_inputs != "bye":
        while not user_inputs:
            print("输入不能为空")
            user_inputs = input(">> user:")
        user_inputs = " ".join(list(user_inputs.replace(" ", "")))
        history.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(user_inputs)))
        output_id = generate(args, history, tokenizer, model, speaker1, speaker2, special_token_ids)
        print(output_id)
        history.append(output_id)
        history = history[-(2*args.max_history+1):]
        print("Bot: ", tokenizer.decode(output_id,skip_special_tokens=True))
        user_inputs = input(">> user:")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("generation interact module")
    parser.add_argument('--checkpoint_path', default='', type=str, help='path to checkpoint')
    parser.add_argument("--top_k", type=int, default=30, help="top k sampling to generate")
    parser.add_argument("--top_p", type=float, default=0.7, help="top p sampling to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="temperature for softmax to generate")
    args = parser.parse_args()
    
    gpt2_model = GPT2Proto.load_from_checkpoint(args.checkpoint_path)
    interact(args, gpt2_model, gpt2_model.tokenizer)