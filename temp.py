from pymongo import MongoClient
from dotenv import load_dotenv
import os
# Connect
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
if not MONGO_URI or not DB_NAME:
    raise RuntimeError("Please set MONGO_URI and DB_NAME in your .env")
mongo = MongoClient(MONGO_URI)
db = mongo[DB_NAME]

coll_info = db.command("listCollections")["cursor"]["firstBatch"]

for coll in coll_info:
    name = coll["name"]
    options = coll.get("options", {})
    validator = options.get("validator", {})
    print(f"\nCollection: {name}")
    print("Schema/Validator:", validator if validator else "No schema defined")
