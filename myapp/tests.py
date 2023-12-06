from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained("t5-large", model_max_length=512)

text = """
18: Saluting Battery Visit
        Cost Per Person: €3: 
        Guide Included cost: Yes: 
        Location: Valletta: 
        Time of Visit hours: 0:30: 
        Associated Places: Valletta: 
 
19: Saluting Battery 1 Gun Fire
        Cost Per Person: €177: 
        Guide Included cost: Yes: 
        Location: Valletta: 
        Time of Visit hours: 0:30: 
        Associated Places: Private: 
"""

preprocessed_text = "generation: " + "Frame answer like itinerary" + text

token_input = tokenizer.encode(preprocessed_text, return_tensors="pt", max_length=512, truncation=True, add_special_tokens=True)

summary_ids = model.generate(
    token_input, min_length=60, max_length=512, num_beams=4, length_penalty=1.5
)

summary = tokenizer.decode(summary_ids[0])

print(summary, "summary==========================")
