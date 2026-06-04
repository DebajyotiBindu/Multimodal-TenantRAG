import chainlit as cl
import os
import sys
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.audit_db import init_db, log_rag_transaction
from src.retrieval import retrieval
from langchain_groq import ChatGroq

from src.loaders import TextLoader, ImageLoader
from src.splitting import Splitter, Embedding
import chromadb

init_db()

VECTOR_DB_DIR = r"D:\mlproject19\Vector_Database"
STATIC_IMAGE_DIR = r"D:\mlproject19\Static"
TEMP_DATA_DIR = r"D:\mlproject19\data"


os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(STATIC_IMAGE_DIR, exist_ok=True)
os.makedirs(TEMP_DATA_DIR, exist_ok=True)


def execute_isolated_ingestion_pipeline(pdf_source_path: str, tenant_uid: int):
    """
    Ingestion Core: Takes the user's isolated PDF, chunks the layout text,
    extracts images, hits the Llama Vision API for metadata captions, 
    embeds via sentence-transformers, and upserts payloads to ChromaDB 
    with explicit tenant filtering metadata tags.
    """
    split_obj = Splitter()
    image_obj = ImageLoader()
    embed_obj = Embedding()


    chunks = split_obj.splitting(file_path=TEMP_DATA_DIR)
    if not chunks:
        chunks = []

    page_info_dict = image_obj.caption_generate(file_path=TEMP_DATA_DIR)
    if not page_info_dict:
        page_info_dict = {}

    new_chunks = [chunk.page_content for chunk in chunks]
    embed_doc = []
    if new_chunks:
        embed_doc = embed_obj.text_embedding(chunks=new_chunks)

    image_metadata = []
    image_keys = list(page_info_dict.keys())
    for key in image_keys:
        image_metadata.append(page_info_dict[key]['captions'])

    if image_metadata:
        image_result = embed_obj.image_embedding(image_metadata=image_metadata)
        for idx, key in enumerate(image_keys):
            page_info_dict[key]['caption_vector'] = image_result[idx]

    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    collection_text = chroma_client.get_or_create_collection(
        name="Text_Collection", 
        embedding_function=None
    )
    if chunks and embed_doc:
        unique_text_ids = [f"text_chunk_{tenant_uid}_{idx}" for idx in range(len(chunks))]
        chunk_metadatas = []
        for chunk in chunks:
            meta = chunk.metadata.copy() if chunk.metadata else {}
            meta['user_id'] = tenant_uid  # Structural isolation filter key injection
            chunk_metadatas.append(meta)

        collection_text.add(
            ids=unique_text_ids,
            embeddings=embed_doc,
            documents=new_chunks,
            metadatas=chunk_metadatas
        )
        print(f"[Chroma Engine] Successfully mapped {len(chunks)} text chunks for tenant_id: {tenant_uid}")

    # 6. Populate Multi-Tenant Image Caption Collection
    collection_image = chroma_client.get_or_create_collection(
        name="Image_Collection", 
        embedding_function=None
    )
    
    img_ids, img_docs, img_embeds, img_metadatas = [], [], [], []
    for img_path, img_data in page_info_dict.items():
        if 'caption_vector' in img_data:
            img_ids.append(f"img_{tenant_uid}_{os.path.basename(img_path)}")
            img_docs.append(img_data['captions'])
            img_embeds.append(img_data['caption_vector'])
            img_metadatas.append({
                'page_number': img_data.get('page_number', 0),
                'user_id': tenant_uid,  # Structural isolation filter key injection
                'filepath': img_data.get('filepath', img_path)
            })

    if img_ids:
        collection_image.add(
            ids=img_ids,
            embeddings=img_embeds,
            documents=img_docs,
            metadatas=img_metadatas
        )
        print(f"[Chroma Engine] Successfully mapped {len(img_ids)} visual assets for tenant_id: {tenant_uid}")


@cl.on_chat_start
async def initialize_session():
    # Construct a unique bound multi-tenant session ID from browser context
    raw_string_id = cl.context.session.id
    unique_int_id = abs(hash(raw_string_id)) % 10**8
    cl.user_session.set("user_id", unique_int_id)
    print(f"Isolated workspace successfully assigned session tenant ID: {unique_int_id}")

    # Request the target document directly via the native UI socket loader widget
    uploaded_files = None
    while uploaded_files is None:
        uploaded_files = await cl.AskFileMessage(
            content="## Multimodal Custom Document RAG\nPlease drop or upload your technical PDF manuscript to build your isolated vector sandbox context.",
            accept=["application/pdf"],
            max_size_mb=35,
            timeout=300
        ).send()

    target_file = uploaded_files[0]
    
    status_bubble = cl.Message(content=f"Parsing, chunking, and embedding `{target_file.name}` into tenant workspace `{unique_int_id}`...")
    await status_bubble.send()

    # Clear structural staging context paths to secure execution against cross-contamination
    if os.path.exists(TEMP_DATA_DIR):
        shutil.rmtree(TEMP_DATA_DIR)
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)

    # Reconstruct the absolute path target structure on disk
    sandbox_pdf_destination = os.path.join(TEMP_DATA_DIR, target_file.name)
    shutil.copy(target_file.path, sandbox_pdf_destination)

    # Offload blocking extraction operations safely to async execution space
    await cl.make_async(execute_isolated_ingestion_pipeline)(sandbox_pdf_destination, unique_int_id)

    status_bubble.content = f"Context online! `{target_file.name}` loaded into database collections under tenant identifier `{unique_int_id}`."
    await status_bubble.update()


@cl.on_message
async def handle_user_query(incoming_message: cl.Message):
    current_uid = cl.user_session.get("user_id")
    user_query = incoming_message.content
    
    # Initialize your native custom search class with multi-tenant filtering constraints
    engine = retrieval(database_path=VECTOR_DB_DIR, query=user_query, user_id=current_uid)

    # Execute matching similarity space lookup inside Chainlit according to step tracking blocks
    async with cl.Step(name="Retrieving context") as context_step:
        text_context, image_context = engine.search()
        status_text = ""

        if text_context:
            status_text += "Found Text on Page " + str(text_context[0].metadata.get("page", "Unknown")) + "\n"
        if image_context:
            status_text += "Found Image on Path " + str(image_context[0].metadata.get("filepath", "Unknown")) + "\n"
        
        context_step.output = status_text

    # Extract target text segments safely
    if text_context:
        text_data = text_context[0].page_content
    else:
        text_data = "No text found"
        
    # Extract structural multimodal captions safely
    if image_context:
        image_data = f"Image path: {image_context[0].metadata.get('filepath', 'Unknown')}\nCaption: {image_context[0].page_content}"
    else:
        image_data = "No image found"

    # Spawn an empty messaging bucket context to interface the async token chunk stream
    ui_message_bubble = cl.Message(content="")
    await ui_message_bubble.send()

    system_prompt = """You are an advanced Multimodal RAG Assistant. Your primary goal is to provide accurate, grounded answers to user questions by synthesizing contextual payloads extracted from local vector databases.

        ### Operational Constraints & Guardrails:
        1. Strict Grounding: Rely ONLY on the provided [TEXT CONTEXT] and [IMAGE CONTEXT] inside the User message to formulate your answer. If the context does not contain enough data to answer the query, state clearly that you cannot find the answer in the source documents. Do not hallucinate or inject external knowledge.
        2. Multimodal Tracking: When referencing information derived from an image caption, explicitly mention the corresponding file path or page location so the user knows which visual asset correlates with your point.
        3. Citation Clarity: If page numbers or source document attributes are present in the text metadata, incorporate them seamlessly into your response (e.g., "According to Page 4...").
        4. Tone: Maintain a professional, highly analytical, and direct engineering tone. Be concise and prioritize structural technical accuracy."""

    human_message_content = f"""Context Datasets:
               
        [TEXT CONTEXT]
        {text_data}

        [IMAGE CONTEXT]
        {image_data}
                
        User Query: {user_query}"""
            
    messages = [
        ("system", system_prompt),
        ("human", human_message_content),
    ]

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    final_response_text = ""
    try:
        # Stream model output tokens natively across established websockets
        async for chunk in llm.astream(messages):
            token = chunk.content
            await ui_message_bubble.stream_token(token)
            final_response_text += token
    except Exception as e:
        print("Inference engine execution interrupted: ", e)
        final_response_text = "Inference failed due to an internal API error."
        await ui_message_bubble.update(content=final_response_text)

    finally:
        # If image contexts were loaded, bind them directly beneath the streaming bubble interface
        if image_context:
            local_image_path = image_context[0].metadata.get("filepath", "Unknown Path")
            if os.path.exists(local_image_path):
                image_element = cl.Image(
                    path=local_image_path,
                    name="Source Diagram",
                    display='inline'
                )
                ui_message_bubble.elements = [image_element]
                await ui_message_bubble.update()

        # Flush the transaction data metrics to your SQLite relational tables via audit_db
        log_rag_transaction(
            user_id=current_uid,
            query=user_query,
            response=final_response_text,
            text_context=text_context,
            image_context=image_context
        )