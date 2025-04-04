import pandas as pd
from langchain_core.documents import Document

def convertdata():
    product_data=pd.read_csv("shoe_dataset_cleaned.csv")

    data=product_data[["product_title","review"]]

    product_list = []

    for index, row in data.iterrows():

        obj = {
            'product_name': row['product_title'],
            'review': row['review']
        }

        product_list.append(obj)

    docs = []
    for entry in product_list:
        metadata = {"product_name": entry['product_name']}
        doc = Document(page_content=entry['review'],metadata=metadata)
        docs.append(doc)
    return docs