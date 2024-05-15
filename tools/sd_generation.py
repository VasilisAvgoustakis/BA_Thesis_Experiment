from transformers import AutoTokenizer, OPTForCausalLM
import torch


def generate_sd_batch(model, tokenizer, device, prompts, min_length, rp):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move all tensors to the right device
    
    outputs = model.module.generate(
        **inputs, 
        max_length=500,  # Determines the maximum length of the generated text
        min_length=300,  # Sets a minimum length for the generated texts
        temperature=0.7,  # Controls randomness: lower values make text less random
        top_k=50,  # The K most likely next words are considered for each step
        top_p=0.9,  # Only the most probable tokens with probabilities that add up to top_p are considered for each step
        repetition_penalty=rp,
        do_sample=True,  # Set to True to return diverse samples
        num_return_sequences=1,  # Number of independently computed samples to generate
        pad_token_id=tokenizer.eos_token_id,  # Ensures that padding is handled correctly
        #pad_token_id=None
    )

    # Decode the output sequences to get the generated text
    generated_batch = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    print("Generated Batch: ", generated_batch)
    #print(len(generated_batch))
    #print("Generated Text: ", generated_text)
    generated_stories =  [' '.join(text.split("\n")[1:] if len(text.split("\n")) > 1 else text) for text in generated_batch]
    #print(len(generated_stories))
    #print("Generated Stories: ", generated_stories)

    # Continue generating if the text is shorter than 300 tokens
    for story in generated_stories:
        story_index = generated_stories.index(story)
        #print(f"Intermittend length: {story_index+1} ", len(tokenizer.encode(story)))
        while len(tokenizer.encode(story)) < min_length:
            
            #print(" Entering Aug loop:")
            #print(f"Intermittend length: {story_index+1} ", len(tokenizer.encode(story)))

            additional_input_ids = tokenizer.encode(story, return_tensors="pt").to(device)
           
            additional_sequences = model.module.generate(
                input_ids=additional_input_ids,
                max_new_tokens=500 - len(additional_input_ids),  # Generate up to 500 tokens in total including input,
                min_length=300,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=rp,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=5,
                eos_token_id=tokenizer.eos_token_id  # Now considering EOS token
            )
            additional_text = tokenizer.decode(additional_sequences[0], skip_special_tokens=True)
            
            if additional_text == story:  # Break if no new text is generated
                break
            story += ' ' + additional_text[len(tokenizer.decode(tokenizer.encode(story), skip_special_tokens=True)):]
            generated_stories[story_index] = story  # Update the story directly in the list
            #print(generated_stories[story_index])
    #token_length = len(tokenizer.encode(story))
    #print("Final length: ", token_length)

    #print(generated_stories)
    return generated_stories


def inference_stories_batch(model, tokenizer, device, prompts, min_length, rp):

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move all tensors to the right device
    
    output_sequences = model.module.generate(
        **inputs,
        max_length=500,  # Determines the maximum length of the generated text
        min_length=300,  # Sets a minimum length for the generated texts
        temperature=0.7,  # Controls randomness: lower values make text less random
        top_k=50,  # The K most likely next words are considered for each step
        top_p=0.9,  # Only the most probable tokens with probabilities that add up to top_p are considered for each step
        repetition_penalty=rp,
        do_sample=True,  # Set to True to return diverse samples
        num_return_sequences=1,  # Number of independently computed samples to generate
        pad_token_id=tokenizer.eos_token_id,  # Ensures that padding is handled correctly
        #pad_token_id=None
    )

    # Decode the output sequences to get the generated text
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in output_sequences]
    
    #print("Generated Text: ", generated_text)
    generated_stories =  [' '.join(text.split("\n")[1:] if len(text.split("\n")) > 1 else text) for text in generated_texts]
    #print("Generated Story: ", generated_story)
    
    
    # Continue generating if the text is shorter than 300 tokens
    for story in generated_stories:
        story_index = generated_stories.index(story)
        print(f"Intermittend length: {story_index+1} ", len(tokenizer.encode(story)))
        while len(tokenizer.encode(story)) < min_length:
            print(" Entering Aug loop:")
            additional_input_ids = tokenizer.encode(story, return_tensors="pt").to(device)
           
            additional_sequences = model.module.generate(
                input_ids=additional_input_ids,
                max_new_tokens=500 - len(additional_input_ids),  # Generate up to 500 tokens in total including input,
                min_length=300,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=rp,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=5,
                eos_token_id=tokenizer.eos_token_id  # Now considering EOS token
            )
            additional_text = tokenizer.decode(additional_sequences[0], skip_special_tokens=True)
            
            if additional_text == story:  # Break if no new text is generated
                break
            story += ' ' + additional_text[len(tokenizer.decode(tokenizer.encode(story), skip_special_tokens=True)):]
            generated_stories[story_index] = story  # Update the story directly in the list

    token_length = len(tokenizer.encode(story))
    print("Final length: ", token_length)

    return generated_stories