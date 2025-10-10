import re 

input_file = "combined_text_file.txt"
output_file = "cleaned_text_file.txt" 

with open(input_file, "r", encoding="utf-8") as f: 
    text = f.read() 

text = re.sub(r"Page\s*\d+(\s*of\s*\d+)?", "", text, flags=re.IGNORECASE)
text = re.sub(r"[•●▪♦■©®™†‡¤✓■]+", " ", text)  
text = re.sub(r"[_\-=]{3,}", " ", text)  
text = re.sub(r"\n{2,}", "\n", text)   
text = re.sub(r"\s{2,}", " ", text)
text = re.sub(r"==== .*? ====", "", text)
text = text.lower().strip() 

with open(output_file, "w", encoding="utf-8") as f: 
    f.write(text) 

print(f"Cleaned text saved in the {output_file}") 

