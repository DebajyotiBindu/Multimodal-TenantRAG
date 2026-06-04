import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from src.loaders import TextLoader,ImageLoader
from typing import List

class Splitter:
    '''
    This class is useful for splitting the data scrapped from the pdfs into small chunks
    for the LLM to access this chunks for retrieval and generation.

    Function:
        splitting- Splits the data into chunks
    Args:
        file_path- string specifying the path to the root directory
    Returns:
        a list of document objects comprising of chunks of content and its metadata
    '''
    def __init__(self):
        pass

    def splitting(self,file_path):
        try:
            doc_obj=TextLoader()
            doc=[] #Saves the texts extracted from the all pdfs inside the root directory

            #Traversing the root directory
            for files in os.listdir(file_path):
                new_file_path=os.path.join(file_path,files)
                doc_obj.init_loader(new_file_path,doc)
            
            text_splitters=RecursiveCharacterTextSplitter( #Splitting th data into chunks
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=[
                    "\n\n",
                    "\n",
                    " ",
                    ""
                ]
            )
            
            chunks=[]
            chunks=text_splitters.split_documents(doc)
            print(chunks[0].metadata)

            print("Chunking Successful")
            return chunks

        except Exception as e:
            print("Chunking can't be done",e)

class Embedding:
    '''
    This is class is useful for embedding or vectorization of the chunks obtained.
    Done so that storing of this embeddings can be done into vector database later.

    Functions:
        text_embedding()
        Args:
            chunks- is a list of just the page content from the documents previously splitted
        Returns:
            The embedded list of page content of the splitted documents
        
        image_embedding()
        Args:
            image_metadata- It is a list comprising of just the captions of the page information dictionary
        Returns:
            a list of embedded image cations vectors
    '''
    def __init__(self):
        self.embeddings=HuggingFaceEmbeddings(  #Using the sentence-transformers model for embedding the text and images
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": True},
        )

    def text_embedding(self,chunks):
        try:
            print("Text Embedding started....")
            doc_result=self.embeddings.embed_documents(chunks)

            #print(doc_result[0])
            print("Text Embedding Done")
            return doc_result
        
        except Exception as e:
            print("Text Embedding is interrupted",e)
    
    def image_embedding(self,image_metadata:List):
        try:
            print("Image Embedding started....")

            image_result=self.embeddings.embed_documents(image_metadata)

            #print(image_result)
            print("Image Embedding Done")
            return image_result
        
        except Exception as e:
            print("Image Embedding is interrupted",e)


def main():
    file_path=r"D:\mlproject19\data"
    split_obj=Splitter()

    chunks=split_obj.splitting(file_path=file_path)       

    image_obj=ImageLoader()
    page_info_dict=image_obj.caption_generate(file_path=file_path)

    embed_obj=Embedding()
    new_chunks=[]
    for chunk in chunks:
        chunk_val=chunk.page_content
        new_chunks.append(chunk_val)
    embed_doc=embed_obj.text_embedding(chunks=new_chunks)

    image_metadata=[]
    for key,value in page_info_dict.items():  
        captions=page_info_dict[key]['captions']
        image_metadata.append(captions)
    
    image_result=embed_obj.image_embedding(image_metadata=image_metadata)
    i=0
    for key,value in page_info_dict.items():  #Mapping of corresponding image caption embeddings to their pre-specified position in dictionary
        page_info_dict[key]['caption_vector']=image_result[i]
        i=i+1

    #first_elem=next(iter(page_info_dict.items())) #Accesses the the first key-value pair from the dictionary
    #print(first_elem)

    return page_info_dict,embed_doc,chunks 

if __name__=="__main__":
    page_info_dict,embed_doc,chunks=main()
    
    print(len(page_info_dict))
    print(len(embed_doc))
    print(chunks[0])