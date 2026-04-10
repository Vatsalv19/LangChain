from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader


loader = DirectoryLoader(
path = 'books',
glob = '*.pdf',
show_progress = True,
loader_cls = PyPDFLoader
)

docs = loader.load()
print(docs[0].page_content)



# lazy loading is a technique where the documents are loaded only when they are needed, rather than loading all the documents at once. This can be useful when dealing with large collections of documents, as it can save memory and improve performance.