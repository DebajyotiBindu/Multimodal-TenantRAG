import os
from langchain_chroma import Chroma
from splitting import Splitter,Embedding
from loaders import ImageLoader
import chromadb

class DataBase:
    '''
    This class is useful for the creation of vector database and storing the vectorized chunks into it.
    This is done to facilitate the retrieval process of the RAG system.
    Functions:
        text_vector()- This function is useful for storing the vectorized text chunks into the vector database.
        Args:
            file_path- string specifying the path to the root directory
            user_id- integer specifying the user id for which the vector database is being created.

        image_vector()- This function is useful for storing the vectorized image chunks into the vector database.
        Args:
            file_path- string specifying the path to the root directory
            user_id- integer specifying the user id for which the vector database is being created.

    '''
    def __init__(self,file_path,user_id):
        self.client=chromadb.PersistentClient(path=file_path)
        self.user_id=user_id
        self.page_info_dict,self.embed_doc,self.chunks=details()
    
    def text_vector(self):
        try:
            collection1=self.client.get_or_create_collection(
                name="Text_Collection",
                embedding_function=None
            )

            print("Text Collection has been initiated....")

            unique_id=[f"text_chunk_{id}" for id in range(len(self.chunks))]
            text=[chunk.page_content for chunk in self.chunks]
            page_number=[chunk.metadata.get('page') for chunk in self.chunks]
            chunk_metadatas=[]
            for chunk in self.chunks:
                meta=chunk.metadata.copy()
                meta['user_id']=self.user_id
                chunk_metadatas.append(meta)

            collection1.add(
                ids=unique_id,
                embeddings=self.embed_doc,
                documents=text,
                metadatas=chunk_metadatas
            )
            print("Text Collection has been made")
            return 
        
        except Exception as e:
            print("Text Collection can't be made",e)

    def image_vector(self):
        try:
            collection2=self.client.get_or_create_collection(
                name="Image_Collection",
                embedding_function=None
            )
            
            print("Image Collection has been initiated....")
            id=[]
            data=[]
            data_embed=[]
            metadata=[]
            for image_path,image_data in self.page_info_dict.items():
                id.append(image_path)
                data.append(image_data['captions'])
                data_embed.append(image_data['caption_vector'])
                metadata.append(
                    {
                        'page_number':image_data['page_number'],
                        'user_id':self.user_id,
                        'filepath':image_data['filepath']
                    }
                )
            if id:
                collection2.add(
                    ids=id,
                    embeddings=data_embed,
                    documents=data,
                    metadatas=metadata
                )
                print("Image Collection has been made")
            else:
                print("No image found")

            return 
        
        except Exception as e:
            print("Image Collection can't be made",e)

'''
The details function is used to extract the text and image data from the root directory, generate captions for the images,
and embed both the text and image data. It returns a dictionary containing the image metadata and their corresponding caption vectors, 
the embedded text chunks, and the original text chunks. 
This function is called in the __init__ method of the DataBase class to initialize the necessary data for creating the vector database.
'''
def details():
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

    first_elem=next(iter(page_info_dict.items())) #Accesses the the first key-value pair from the dictionary
    print(first_elem)

    return page_info_dict,embed_doc,chunks 

def main():
    file_path=r"./Vector_Database"
    os.makedirs(file_path,exist_ok=True)
    database_obj=DataBase(file_path=file_path,user_id=1)

    database_obj.text_vector()
    database_obj.image_vector()
    return 

if __name__=="__main__":
    main()