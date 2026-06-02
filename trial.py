import pandas as pd
import requests
import os

# 1. Load the Excel file
file_path = 'test.xlsx'
df = pd.read_excel(file_path)

# 2. Specify the column with URLs and create a save folder
url_column = 'image_url' 
# Save inside the current project directory
save_folder = os.path.join(os.getcwd(), 'images')
os.makedirs(save_folder, exist_ok=True)
print(f"Saving images to: {save_folder}")

# 3. Loop through URLs and download
for index, url in enumerate(df[url_column]):
    if pd.isna(url):  # Skip empty cells
        continue
    
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            extension = url.split('.')[-1].split('?')[0]  
            filename = f"image_{index}.{extension}"
            full_path = os.path.join(save_folder, filename)
            
            with open(full_path, 'wb') as f:
                f.write(response.content)
                print(f"Downloaded: {filename}")
        else:
            print(f"Failed {url}: Status {response.status_code}")
            
    except Exception as e:
        print(f"Error downloading {url}: {e}")
