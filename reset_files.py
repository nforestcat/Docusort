import os
import shutil

def reset_files():
    input_dir = 'input'
    output_dir = 'output/classified'
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        
    # Collect all PDF files from output/classified (recursive)
    pdfs = []
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdfs.append(os.path.join(root, file))
    
    # Also collect from input (if any)
    for file in os.listdir(input_dir):
        if file.lower().endswith('.pdf'):
            pdfs.append(os.path.join(input_dir, file))
            
    print(f"Total PDFs found: {len(pdfs)}")
    
    temp_dir = 'temp_reset'
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Move and rename
    for i, pdf_path in enumerate(pdfs, 1):
        shutil.copy2(pdf_path, os.path.join(temp_dir, f"{i}.pdf"))
        
    # Clear and move back
    for f in os.listdir(input_dir):
        if f.lower().endswith('.pdf'): os.remove(os.path.join(input_dir, f))
    
    for f in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, f), os.path.join(input_dir, f))
        
    shutil.rmtree(temp_dir)
    print("✅ Reset complete. Files are in 'input/' named 1.pdf, 2.pdf, etc.")

if __name__ == "__main__":
    reset_files()
