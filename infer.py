from transformers import LlamaTokenizer
import torch
from modeling.modeling_moe import LlamaForCausalLM
from modeling.configuration_moe import LlamaConfig

def generate(tokenizer, model, text):
    inputs = [text]
    tokens = tokenizer(inputs,return_tensors="pt")
    input_ids = tokens.input_ids.cuda()
    generate_ids = model.generate(inputs=input_ids,
                num_beams=1, 
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=256, top_p=0.9, min_p=0.1, temperature=1.0, do_sample=True, no_repeat_ngram_size=2)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = f"{text}{[outputs[i][len(inputs[i]):] for i in range(len(outputs))][0]}"
    return response    


if __name__ == "__main__":
    # model_path = 'mistralai/Mixtral-8x7B-v0.1'
    # model_path = 'meta-llama/Llama-2-7b-hf'
    model_path = 'AnLan577/Dynamic_MoE'
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token
    print(tokenizer)

    model_config = LlamaConfig.from_pretrained(model_path)
    # model_config.num_experts = 4
    
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        from_tf=False,
        config=model_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    ) 
       
    # print("[DEBUG] Model Config:", model_config)
    # print(model.eval())
    model.eval()

    response = generate(tokenizer, model, 'What is the highest mountain in the world?')
    print(response)
    
