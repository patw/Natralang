# Basic flask stuff for building http APIs and rendering html templates
from flask import Flask, render_template, redirect, url_for, request, session, jsonify

# Bootstrap integration with flask so we can make pretty pages
from flask_bootstrap import Bootstrap

# Flask forms integrations which save insane amounts of time
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, TextAreaField, IntegerField, FloatField
from wtforms.validators import DataRequired

# Basic python stuff
import os
import re
import json
import functools

# Mongo stuff
import pymongo
from bson import ObjectId

# Some nice formatting for code
import misaka

# Import OpenAI, Azure and Mistral libraries
from openai import OpenAI
from openai import AzureOpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Nice way to load environment variables for deployments
from dotenv import load_dotenv
load_dotenv(override=True)

# Create the Flask app object
app = Flask(__name__)

# Session key
app.config['SECRET_KEY'] = os.environ["SECRET_KEY"]
app.config['SESSION_COOKIE_NAME'] = 'natralang'

# API Key for app to serve API requests to clients
# API_KEY = os.environ["API_KEY"]

# User Auth
users_string = os.environ["USERS"]
users = json.loads(users_string)

# Define a decorator to check if the user is authenticated
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if users != None:
            if session.get("user") is None:
                return redirect(url_for('login'))
        return view(**kwargs)        
    return wrapped_view

# Determine which LLM/embedding service we will be using
# this could be llamacpp, mistral or openai
service_type = os.environ["SERVICE"]

# Configure the various llm/embedding services
if "MISTRAL_API_KEY" in os.environ:
    mistral_client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    model_name = os.environ["MODEL_NAME"]
    DEFAULT_SCORE_CUT = 0.82
    DEFAULT_VECTOR_DIMENSIONS = 1024

if "OPENAI_API_KEY" in os.environ:
    oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_name = os.environ["MODEL_NAME"]
    embed_model_name = "text-embedding-3-small"
    DEFAULT_SCORE_CUT = 0.7
    DEFAULT_VECTOR_DIMENSIONS = 1536

if "AZURE_API_KEY" in os.environ:
    oai_client = AzureOpenAI(api_key=os.environ["AZURE_API_KEY"], api_version="2023-12-01-preview", azure_endpoint=os.environ["AZURE_ENDPOINT"])
    model_name = os.environ["MODEL_NAME"]
    embed_model_name = os.environ["AZURE_EMBED_MODEL"]
    DEFAULT_SCORE_CUT = 0.85
    DEFAULT_VECTOR_DIMENSIONS = 1536

# Some handy defaults that control the app
DEFAULT_QUERY_SYSTEM_MESSAGE = "You produce valid Mongo aggregation queries based on the collection descriptions and the users question.  You always output in json"
DEFAULT_QUERY_PROMPT = "Mongo Collections:\n{collections}\n\nUsing the above collections, create a mongo aggregation that can answer '{question}'. ONLY OUTPUT A VALID MONGO AGGREGATION PIPELINE IN JSON AND NOTHING ELSE!"

DEFAULT_ANSWER_SYSTEM_MESSAGE = "You provide detailed summaries to answer the users question."
DEFAULT_ANSWER_PROMPT = "Query results:\n{results}\n\nUsing the above results from a mongo query, answer the users question: '{question}'"

DEFAULT_TEMPERATURE = 0.1
DEFAULT_LIMIT = 3
DEFAULT_CANDIDATES = 100
DEFAULT_SOURCES_PER_PAGE = 10
DEFAULT_TEXT_SCORE_CUT = 3.0 # This is a hack, reranking is better!

# Connect to mongo using environment variables
client = pymongo.MongoClient(os.environ["MONGO_CON"])
db = client[os.environ["MONGO_DB"]]
col = db["sources"]  # sources collection stores the data sources we want to query

# If this is the first time we've run the app we will test to see if the collections exist
# and if they do not, we create them and the lexical and vector search indexes
try:
    # Collection does not exist, create it
    db.create_collection('sources')

    # Create the text/vector search index automatically
    command = {
    "createSearchIndexes": "sources",
    "indexes": [
        {
            "name": 'default',
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "description": {
                            "type": "string"
                        },
                        "sample": {
                            "type": "string"
                        }
                    }
                },
                "analyzer": "lucene.english"
            },
        },
        {
            "name": 'vector',
            "type":"vectorSearch",
            "definition": {
                "fields": [
                    {
                        "path":"description_embedding",
                        "type": "vector",
                        "numDimensions": DEFAULT_VECTOR_DIMENSIONS,
                        "similarity": "cosine"
                    }
                ]
            },
        }
    ]}
    db.command(command)
except:
    # We've already configured the collection/search indexes before just keep going
    # Yes we intentially want this to fail on startup in most cases.
    pass

# Make it pretty because I can't :(
Bootstrap(app)

# A form for asking your external brain questions
class QuestionForm(FlaskForm):
    question = StringField('Question 💬', validators=[DataRequired()])
    k = IntegerField("K Value", validators=[DataRequired()])
    collections = FloatField("Number of Collections", validators=[DataRequired()])
    score_cut = FloatField("Score Cut Off", validators=[DataRequired()])
    submit = SubmitField('Submit')
       
# A form for testings semantic search of the chunks
class VectorSearchForm(FlaskForm):
    question = StringField('Question 💬', validators=[DataRequired()])
    k = IntegerField("K Value", validators=[DataRequired()])
    score_cut = FloatField("Score Cut Off", validators=[DataRequired()])
    submit = SubmitField('Search')
  
# A form for reviewing and saving the summarized facts
class SourcesSearchForm(FlaskForm):
    query = StringField('Search Sources', validators=[DataRequired()])
    submit = SubmitField('Search')

# A form to edit the facts
class SourceEditForm(FlaskForm):
    db_name = StringField('Database Name', validators=[DataRequired()])
    col_name = StringField('Collection Name', validators=[DataRequired()])
    description = TextAreaField('Description of Collection (be detailed)', validators=[DataRequired()])
    sample = TextAreaField('Sample Document', validators=[DataRequired()])
    save = SubmitField('Save')

# Amazing, I hate writing this stuff
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Call OpenAI's new embedder (1536d)
def embed_oai(text):
    text = text.replace("\n", " ")
    return oai_client.embeddings.create(input = [text], model=embed_model_name).data[0].embedding

# Call mistral's embedder (1024d)
def embed_mistral(text):
    text = text.replace("\n", " ")
    return mistral_client.embeddings(model="mistral-embed", input=[text]).data[0].embedding

# Query mistral models
def llm_mistral(prompt, system_message, temperature):
    messages = [ChatMessage(role="system", content=system_message), ChatMessage(role="user", content=prompt)]
    response = mistral_client.chat(model=model_name, temperature=temperature, messages=messages)
    return response.choices[0].message.content

# Query OpenAI models
def llm_oai(prompt, system_message, temperature):
    messages = [ChatMessage(role="system", content=system_message), ChatMessage(role="user", content=prompt)]
    response = oai_client.chat.completions.create(model=model_name, temperature=temperature, messages=messages)
    return response.choices[0].message.content

# Use whichever embedder makes sense for the configured service
def embed(text):
    if service_type == "openai" or service_type == "azure":
        return embed_oai(text)
    if service_type == "mistral":
        return embed_mistral(text)

# Determine which LLM we should call depending on what's configured
def llm(user_prompt, system_message, temperature=DEFAULT_TEMPERATURE):
    if service_type == "openai" or service_type == "azure":
        return llm_oai(user_prompt, system_message, temperature)
    if service_type == "mistral":
        return llm_mistral(user_prompt, system_message, temperature)

# Get all data sources
def get_sources(skip,limit):
    result = col.find().skip(skip).limit(limit).sort([("date", -1)])
    return result

# Return the count of facts in the system
def count_sources():
    return  col.count_documents({})
        
# Retrieve data sources based on the prompt
def search_sources_vector(prompt, candidates, limit, score_cut):
    
    # Get the embedding for the prompt first
    vector = embed(prompt)

    # Build the Atlas vector search aggregation
    vector_search_agg = [
        {
            "$vectorSearch": { 
                "index": "vector",
                "path": "description_embedding",
                "queryVector": vector,
                "numCandidates": candidates, 
                "limit": limit
            }
        },
        {
            "$project": {
                "db_name": 1,
                "col_name": 1,
                "description": 1,
                "sample": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        },
        {
            "$match": {
                "score": { "$gte": score_cut }
            }
        }
    ]

    # Connect to chunks, run query, return results
    result = col.aggregate(vector_search_agg)
    # Return this as a list instead of a cursor
    return result

# Get chunks based on text search 
def search_sources_text(prompt, limit, text_score_cut):
    # Build the Atlas vector search aggregation
    text_search_agg = [
        {   
            "$search": {
                "text": {
                    "path": ["sample", "description"],
                    "query": prompt
                }
            }   
        },
        {
            "$limit": limit
        },
        {
            "$project": {
                "db_name": 1,
                "col_name": 1,
                "description": 1,
                "sample": 1,
                "score": {"$meta": "searchScore"}
            }
        },
        {
            "$match": {
                "score": { "$gte": text_score_cut }
            }
        }
    ]

    # Connect to chunks, run query, return results
    result = col.aggregate(text_search_agg)
    # Return this as a list instead of a cursor
    return result

# The default question view
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():

    # Question form for the external brain
    form = QuestionForm()
    form.k.data = DEFAULT_CANDIDATES
    form.score_cut.data = DEFAULT_SCORE_CUT
    form.collections.data = DEFAULT_LIMIT

    # Store results
    result_data = {}
    result_data["databases"] = []
    result_data["collections"] = []

    # If user is prompting send it
    if form.validate_on_submit():

         # Get the form variables
        form_result = request.form.to_dict(flat=True)
        q = form_result["question"]
        result_data["question"] = q

        # Use vector search to find the data sources relevant to the question
        sources = search_sources_vector(q, int(form_result["k"]), int(form_result["collections"]), float(form_result["score_cut"]))

        # Convert the sources into a string blob for the prompt
        source_blob = ""
        for source in sources:
            result_data["collections"].append({"database": source["db_name"], "collection": source["col_name"]})
            source_blob += "Collection Name: " + source["col_name"] + "\n"
            source_blob += "Description: " + source["description"] + "\n"
            source_blob += "Example Document: " + source["sample"] + "\n\n"

        # We don't have any data sources :(
        if source_blob == "":
            result_data["llm_answer"] = "I couldn't find any data sources that could answer this question."
            # Spit out the template - without data :(
            return render_template('index.html', result_data=result_data, form=form)


        # Assemble the final prompt
        prompt = DEFAULT_QUERY_PROMPT.format(collections=source_blob, question=q)
        result_data["prompt"] = prompt

        # Generate mongo query using the LLM and our assembled prompt
        # Temperature is VERY low here, we want no creativity in this process
        llm_generated_query = llm(prompt, DEFAULT_QUERY_SYSTEM_MESSAGE, 0.1)

        # Clean up weird stuff the LLM generates
        llm_generated_query = llm_generated_query.replace("\\", "")  # Why do you add backslashes to things

        try:
            result_data["llm_generated_query"] = json.loads(llm_generated_query) # convert to json
        except:
            result_data["llm_generated_query"] = llm_generated_query

        # Now the moment of truth... does our generated query execute and return data?
        result_data["query_output"] = []
        good_query = True
        for collection in result_data["collections"]:
            query_db = client[collection["database"]]
            query_col = query_db[collection["collection"]]
            try:
                query_result = query_col.aggregate(result_data["llm_generated_query"])
                for doc in query_result:
                    result_data["query_output"].append(doc)
            except:
                result_data["query_output"].append("Query failed to execute.")
                good_query = False

        # Lets go one extra step here and have the LLM interpret the answer, but limit to the first 50 results
        if good_query:
            answer_prompt = DEFAULT_ANSWER_PROMPT.format(results=result_data["query_output"][:30], question=result_data["question"])
            result_data["llm_answer"] = llm(answer_prompt, DEFAULT_ANSWER_SYSTEM_MESSAGE, 0.7)
        else:
            result_data["llm_answer"] = "Query failed to execute."

        # Spit out the template - with data!
        return render_template('index.html', result_data=result_data, form=form)
    
    # Spit out the template - without data :(
    return render_template('index.html', result_data=result_data, form=form)

# Regenerate the chunks!
@app.route('/sources', methods=['GET', 'POST'])
@login_required
def sources():
    form = SourcesSearchForm()
    source_count = count_sources()

    # For paginating the display
    page = request.args.get('page')
    if not page:
        page = 0
    else:
        page = int(page)

    # Make sure the page doesn't go too low or too high
    if page < 0:
        page = 0
    if page >= source_count / DEFAULT_SOURCES_PER_PAGE:
        page = int(source_count / DEFAULT_SOURCES_PER_PAGE)

    if form.is_submitted():
        form_result = request.form.to_dict(flat=True)
        sources = search_sources_text(form_result["query"], DEFAULT_SOURCES_PER_PAGE, 0.0)
    else:
        skip = page * DEFAULT_SOURCES_PER_PAGE
        limit = DEFAULT_SOURCES_PER_PAGE
        sources = get_sources(skip, limit)
    return render_template('sources.html', page=page, form=form, sources=sources, source_count=source_count)

# Search chunks
@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    sources = []
    form = VectorSearchForm(k=DEFAULT_CANDIDATES, score_cut=DEFAULT_SCORE_CUT)

    # Regen the chunks based on settings
    if form.is_submitted():
        form_result = request.form.to_dict(flat=True)
        # Search chunks using vector search
        prompt = form_result["question"]
        candidates = int(form_result["k"])
        score_cut = float(form_result["score_cut"])
        sources = search_sources_vector(prompt, candidates, 10, score_cut)
    # Render the search template
    return render_template('search.html', sources=sources, form=form)


# This fact is wrong and will be corrected harshly
@app.route('/new', methods=['GET', 'POST'])
@login_required
def source_new():

    # Data source edit
    form = SourceEditForm()

    # Store a new record
    if form.is_submitted():
        form_result = request.form.to_dict(flat=True)
        form_result.pop('csrf_token')
        form_result.pop('save')
        form_result["description_embedding"] = embed(form_result["description"])
        col.insert_one(form_result)
        return redirect(url_for('sources'))
    
    # Render the form
    return render_template('edit.html', form=form)

# This fact is wrong and will be corrected harshly
@app.route('/edit/<id>', methods=['GET', 'POST'])
@login_required
def source_edit(id):

    # Data source edit form
    form = SourceEditForm()

    if id:
        # Update an existing record
        if form.is_submitted():
            form_result = request.form.to_dict(flat=True)
            form_result.pop('csrf_token')
            form_result.pop('save')
            form_result["description_embedding"] = embed(form_result["description"])
            col.update_one({'_id': ObjectId(id)}, {'$set': form_result})
            return redirect(url_for('sources'))
        # Load an existing record
        else:
            source_record = col.find_one({'_id': ObjectId(id)})
            form.db_name.data = source_record["db_name"]
            form.col_name.data = source_record["col_name"]
            form.description.data = source_record["description"]
            form.sample.data = source_record["sample"]

    # Render the form
    return render_template('edit.html', form=form)

# Delete a source
@app.route('/delete/<id>')
@login_required
def source_delete(id):    
    col.delete_one({'_id': ObjectId(id)})
    return redirect(url_for('sources'))

# Login/logout routes that rely on the user being stored in session
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.username.data in users:
            if form.password.data == users[form.username.data]:
                session["user"] = form.username.data
                return redirect(url_for('index'))
    return render_template('login.html', form=form)

# We finally have a link for this now!
@app.route('/logout')
def logout():
    session["user"] = None
    return redirect(url_for('login'))
