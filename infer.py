from transformers import AutoTokenizer
import torch
from modeling.modeling_moe import MoEForCausalLM
from modeling.configuration_moe import MoEOPTConfig



# def generate(tokenizer, model, text):
#     inputs = [text]
#     tokens = tokenizer(inputs,return_tensors="pt")
#     input_ids = tokens.input_ids.cuda()
#     print("input_ids:", input_ids)
#     print("input_ids shape:", input_ids.shape)
#     print("input_ids device:", input_ids.device)
#     print("model.device:", next(model.parameters()).device)
#     torch.set_printoptions(precision=4, sci_mode=False)
#     generate_ids = model.generate(inputs=input_ids,
#                 num_beams=1, 
#                 bos_token_id=tokenizer.bos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#                 pad_token_id=tokenizer.pad_token_id,
#                 max_new_tokens=256,top_p=0.9, temperature=1.0, do_sample=True)
#     outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#     response = [outputs[i][len(inputs[i]):] for i in range(len(outputs))][0]
#     return response    
    
def generate(tokenizer, model, prompt: str, max_length=50):
    device = next(model.parameters()).device
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Optional: print shapes/devices
    print(f"[DEBUG] input_ids.shape: {input_ids.shape}, device: {input_ids.device}")
    
    # Manually patch logits via logit processor or override forward (recommended only for testing)
    with torch.no_grad():
        try:
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                do_sample=True,         # Needed for multinomial
                temperature=1.0,         # Try setting to 0.7 or 1.0 (avoid too low)
                top_p=0.95,
                output_scores=True,
                return_dict_in_generate=True,
            )
        except RuntimeError as e:
            print(f"[ERROR] RuntimeError: {e}")
            print("[DEBUG] Try enabling CUDA_LAUNCH_BLOCKING=1 and logging logits before multinomial.")
            raise

    return tokenizer.decode(output_ids.sequences[0], skip_special_tokens=True)

if __name__ == "__main__":
    # model_path = 'AnLan577/Dynamic_MoE'
    # model_path = 'meta-llama/Llama-2-7b-hf'
    model_path = 'facebook/opt-350m'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token

    model_config = MoEOPTConfig.from_pretrained(model_path,trust_remote_code=True)
    model = MoEForCausalLM.from_pretrained(
        model_path,
        from_tf=False,
        config=model_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).cuda()    
    model.eval() 
    
    response = generate(tokenizer, model, 'The highest mountain in the world is')
    print(response)
    
