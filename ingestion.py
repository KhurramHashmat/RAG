import pymupdf 
import os 

pdf_dir = "AIPDFS" 
output_file = "combined_text_file.txt" 

with open("combined_text_file.txt", "w", encoding="utf-8") as out_f: 
    
    for filename in sorted(os.listdir(pdf_dir)): 
        if filename.lower().endswith(".pdf"): 
            file_path = os.path.join(pdf_dir, filename) 

            print(f"Extracting from {filename}") 

            pdf_doc = pymupdf.open(file_path) 

            out_f.write(f"\n\n==== {filename} ====\n\n")

            for page_num, page in enumerate(pdf_doc, start=1): 
                text = page.get_text("text")
                out_f.write(text)
                out_f.write("\n")

            pdf_doc.close() 

print(f"All text extracted and saved in {output_file}") 



