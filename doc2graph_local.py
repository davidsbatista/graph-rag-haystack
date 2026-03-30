import csv
import json
from collections import defaultdict
from typing import List

from tqdm import tqdm

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret, ComponentDevice, Device
from neo4j import GraphDatabase

from prompts import (
    ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_SYSTEM_HF,
    ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_USER_HF,
)


def read_documents(file: str) -> List[Document]:
    with open(file, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader, None)  # skip the headers
        documents = []
        for row in reader:
            category = row[0].strip()
            title = row[2].strip()
            text = row[3].strip()
            documents.append(Document(content=text, meta={"category": category, "title": title}))

    return documents


def build_pipeline():
    """
    Build a local Haystack pipeline using Phi-3-mini fine-tuned for graph extraction.

    Requires HF_API_TOKEN to be set as an environment variable.
    """
    generator = HuggingFaceLocalChatGenerator(
        token=Secret.from_env_var("HF_API_TOKEN"),
        task="text-generation",
        model="EmergentMethods/Phi-3-mini-4k-instruct-graph",
        # Uncomment the line below to use Apple Silicon GPU acceleration:
        # device=ComponentDevice.from_single(Device.mps()),
        device=ComponentDevice.from_single(Device.cpu()),
        generation_kwargs={
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
    )

    prompt = ChatPromptBuilder()
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", prompt)
    pipeline.add_component("generator", generator)
    pipeline.connect("prompt_builder", "generator")

    messages = [
        ChatMessage.from_system(ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_SYSTEM_HF),
        ChatMessage.from_user(ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_USER_HF),
    ]

    return pipeline, messages


def load_data_to_neo4j(nodes, edges):
    driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "neo4j"))

    with driver.session() as session:
        for node in nodes:
            session.run(
                "MERGE (n:Node {id: $id}) SET n.name = $name, n.type = $type, n.detailed_type = $detailed_type",
                id=node['id'],
                name=node['name'],
                type=node['type'],
                detailed_type=node['detailed_type']
            )

        for edge in edges:
            session.run(
                """
                MATCH (a:Node {id: $source}), (b:Node {id: $target})
                MERGE (a)-[r:RELATIONSHIP {description: $description}]->(b)
                """,
                source=edge['source'],
                target=edge['target'],
                rel_type=edge['description'],
                description=edge['description']
            )

    driver.close()


def extract():
    documents = read_documents("bbc-news-data.csv")
    docs = [doc for doc in documents if doc.meta['category'] == 'business']

    pipeline, messages = build_pipeline()
    extracted_graph = defaultdict(list)

    for d in tqdm(docs[0:10], desc="Processing documents"):
        data = {"prompt_builder": {"template_variables": {"input_text": d.content}, "template": messages}}
        result = pipeline.run(data=data)
        result_json = json.loads(result['generator']['replies'][0].content.strip())
        extracted_graph["nodes"].extend(result_json['nodes'])
        extracted_graph["edges"].extend(result_json['edges'])

    load_data_to_neo4j(extracted_graph["nodes"], extracted_graph["edges"])


if __name__ == "__main__":
    extract()
