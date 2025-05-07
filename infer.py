from transformers import AutoTokenizer
import torch
from modeling.modeling_moe import MoEForCausalLM
from modeling.configuration_moe import MoEConfig

def generate(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    generate_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=1, 
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=256,
                top_p=0.9, 
                temperature=1.0, 
                do_sample=True)
    # outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # response = [outputs[i][len(inputs[i]):] for i in range(len(outputs))][0]
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
    return outputs[0]   
    
    
if __name__ == "__main__":
    model_path = "AnLan577/Dynamic_MoE"
    # model_path = 'meta-llama/Llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token

    model_config = MoEConfig.from_pretrained(model_path,trust_remote_code=True)
    model = MoEForCausalLM.from_pretrained(
        model_path,
        from_tf=False,
        config=model_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )    
    model.eval() 

    response = generate(tokenizer, model, 'The highest mountain in the world is')
    print(response)
    
