import os
from langchain_community.document_loaders import PyPDFLoader
import fitz
from tqdm import tqdm 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root) #This links/adds COMPONENT folder with src folder directory

from COMPONENTS.captions import encode_image,caption_generator

class TextLoader:
    '''
    This class is useful for loading the texts from the pdf document provided.
    This returns a list of texts extracted from the pdfs present in the path provided.

    Function: init_loader()
    Args: 
        path- A string of the path where the pdf of located.
        doc- This is a list for storage,(that is intended to be passed on by reference) 
        storing all the textual information loaded.
    '''
    def __init__(self):
        pass
    
    def init_loader(self,path:str,doc):

        #Initializing the loader
        try:
            loader=PyPDFLoader(
                file_path=path, 
                headers=None,
                password=None,
                mode="page",
                pages_delimiter="",
                extraction_mode="layout"
            )

            doc_loader=loader.load()

            for val in doc_loader:
                doc.append(val)
            print(doc[0].page_content[:100])
            print(doc[0].metadata)

            print("Text Extracted")
            return 
        
        except Exception as e:
            print(e)

class ImageLoader:
    '''
    This class is useful for loading/extracting and saving the images from the pdf document provided.
    This returns a dictionary comprising of the reference, file path 
    and the page number of the image extracted thats is the entire metadata of all the images.

    Function: 
        extract()- Extracts all the images from the all PDFs and returns
        a dictionary having metadata regarding image

    Args: 
        workdir- A string of the path where the pdf of located.
        savedir- The new directory path created where the extracted images will be stored.
    
    Function: 
        caption_generate()- Generates the captions of the images saved in the destined directory
        and returns the modified dictionary containing image metadata added with the captions of 
        all the images.
    Args:
        file_path- A string representing the file path of the data's root directory containing all PDFs
    '''
    def __init__(self):
        pass 

    def extract(self,workdir:str,savedir:str):
        try:
            page_info={}
            for each_path in os.listdir(workdir):
                if ".pdf" in each_path:
                    doc = fitz.open(os.path.join(workdir, each_path))

                    #Extracting images page-wise
                    for i in tqdm(range(len(doc)), desc="pages"):
                        for img in doc.get_page_images(i):
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            file_path=os.path.join(savedir, "%s_p%s-%s.png" % (each_path[:-4], i, xref))
                            pix.save(file_path)

                            #Saving the image and its page number
                            page_info[file_path]={
                                "filepath":file_path,
                                "page_number":i,
                                "xref":xref
                            } 
                doc.close()
            print("Image extracted and saved")
            return page_info

        except Exception as e:
            print("Image can't be extracted.",e)
    
    def caption_generate(self,file_path):
        try:
            image_path=r"D:\mlproject19\Static"
            os.makedirs(image_path,exist_ok=True)

            page_info_dict=self.extract(file_path,savedir=image_path)

            for image in os.listdir(image_path):
                if image.endswith('.png'):
                    base64_image=encode_image(os.path.join(image_path,image))
                    message=caption_generator(base64_image=base64_image)
                    img_file_path=os.path.join(image_path,image)

                    if img_file_path in page_info_dict:
                        page_info_dict[img_file_path]['captions']=message #Modifying the image metadata
            
            print("Captions Generated and stored sucessfully")
            return page_info_dict 
        
        except Exception as e:
            print("Captions can't be generated",e)
                
def main():
    file_path=r"D:\mlproject19\data"
    doc_obj=TextLoader()
    doc=[] #Saves the texts extracted from the all pdfs inside the root directory

    #Traversing the root directory
    for files in os.listdir(file_path):
        new_file_path=os.path.join(file_path,files)
        doc_obj.init_loader(new_file_path,doc)
    
    image_obj=ImageLoader()
    page_info_dict=image_obj.caption_generate(file_path=file_path)
    return page_info_dict


if __name__=="__main__":
    page_info_dict=main()
    print(page_info_dict)    