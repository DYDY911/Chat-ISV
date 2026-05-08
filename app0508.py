import os
import time
import json
import gradio as gr
from neo4j import GraphDatabase 
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from pyvis.network import Network
import html
import tempfile

# 🚀 Local Vector DB and Embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

# ==========================================
# 1. Read Database Credentials
# ==========================================
NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "") 
NEO4J_DATABASE_NAME = "e554e789" 

# ==========================================
# 2. Database Initialization (LangChain RAG)
# ==========================================
try:

    neo4j_driver = GraphDatabase.driver(
        NEO4J_URI, 
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        keep_alive=True,              
        max_connection_lifetime=180   
    )


    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE_NAME
    )
    graph.refresh_schema()
    
    print("✅ LangChain Graph RAG connected successfully!")

except Exception as e:
    print(f"⚠️ Graph connection skipped (running in Vector/Web fallback mode): {e}")
    graph = None 
    neo4j_driver = None
# ==========================================
# 2.5 🚀 Load Local Vector DB (Direct Root Path)
# ==========================================
try:
    print("🚀 Loading local environmental vector database (CPU mode)...")
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 🎯 Point directly to the current root directory "." 
    faiss_path = "."
    
    # First check if both files actually exist in the root directory
    if os.path.exists("index.faiss") and os.path.exists("index.pkl"):
        vectorstore = FAISS.load_local(faiss_path, hf_embeddings, allow_dangerous_deserialization=True)
        print("✅ Local Vector DB loaded successfully from root directory! 3-Tier Architecture Ready!")
    else:
        print("⚠️ Vector DB load failed: 'index.faiss' or 'index.pkl' not found in the root directory.")
        vectorstore = None

except Exception as e:
    print(f"⚠️ Vector DB load failed: {e}")
    vectorstore = None

# ==========================================
# 3. Dynamic Graph Chain Initialization
# ==========================================
def get_graph_chain(api_key: str, base_url: str):
    llm_kwargs = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "openai_api_key": api_key
    }
    if base_url and base_url.strip():
        llm_kwargs["base_url"] = base_url.strip()
        
    llm = ChatOpenAI(**llm_kwargs)

    cypher_template = """You are a Neo4j Cypher expert. Convert the user's question into a Cypher query.
    Use only the provided schema. Do NOT wrap the Cypher query in markdown block formatting (e.g. ```cypher ... ```). 
    Just return the raw Cypher query string.

    [CRITICAL RULES FOR THIS SPECIFIC GRAPH]:
    1. STRICT RELATIONSHIPS: NEVER invent relationship types. For example, a Process always -[:EMITS]-> a VOCSpecies. NEVER use made-up relationships like `[:PARTICIPATES_IN]`. If you don't know the exact relationship, omit it and just query the node directly, or use a generic relationship `-[r]-`.
    2. FUZZY MATCHING & ATTRIBUTES (CRITICAL TYPE CASTING): NEVER use exact matching. ALWAYS use case-insensitive fuzzy matching. Because database property types may vary (some might be arrays or numbers), you MUST wrap properties in `toString()` before using `toLower()`. 
    Example: `WHERE toLower(toString(n.name)) CONTAINS "pmf" OR toLower(toString(n.evidence_span)) CONTAINS "pmf"`.
    3. AVOID OVER-CONSTRAINING: Do NOT chain multiple `AND` conditions on names. If searching for "aromatic hydrocarbons in coking", focus on extracting EVERYTHING related to coking: `MATCH (p:Process)-[:EMITS]->(v:VOCSpecies) WHERE toLower(toString(p.name)) CONTAINS "coking" RETURN...` and let the QA model filter the aromatics later.
    4. RETURN CLEAN DICTIONARIES (CRITICAL): Do NOT return the raw entire node. Construct a clean dictionary in your RETURN statement containing ONLY the entity's name and its detailed text/evidence. 
    Example: `RETURN p.name AS Process, {{name: v.name, details: coalesce(v.evidence_span, "No details")}} AS Data`.
    5. PREVENT TRUNCATION: ALWAYS append "LIMIT 100" to the end of your Cypher queries.
    6. CONTEXT ISOLATION: If the Question contains "--- PREVIOUS CONTEXT ---", you MUST entirely IGNORE that part when generating Cypher. Extract entities ONLY from the "--- CURRENT QUESTION ---".
    7. NO POST-PROCESSING OR CONDITIONAL LOGIC: NEVER use `UNION`, `CASE WHEN`, or `count(*)` to format string responses (e.g., returning 'None recorded' if empty). Your ONLY job is to write a simple MATCH/RETURN query to fetch data. The downstream QA model will handle text formatting and empty states.
    8. SINGLE STATEMENT ONLY (CRITICAL): NEVER return multiple Cypher statements separated by semicolons (`;`). You MUST return exactly ONE single Cypher statement per request. If you need to fetch multiple different entity types (e.g., both VOCs and ControlTech), you MUST use `OPTIONAL MATCH` within a single query, NOT multiple queries.

    Schema:
    {schema}

    Question:
    {question}

    Cypher query:"""
    cypher_prompt = PromptTemplate(template=cypher_template, input_variables=["schema", "question"])
    qa_template = """You are an authoritative environmental engineering and materials science expert.
    Please synthesize a highly professional, fluent English answer strictly based on the provided [Context] retrieved from the domain knowledge graph.
    
    [CRITICAL RULES - READ CAREFULLY]:
    1. EXTREME GREEDY EXTRACTION: You MUST extract EVERYTHING relevant from the raw graph outputs. Do not ignore ANY specific chemical names, emission factors, or specific concentrations.
    2. STRICT UNIT PRESERVATION: Format numerical data logically (e.g., "11.14 g/t").
    3. ADAPTIVE SCOPE FOCUS (CRITICAL): Analyze the user's question. 
       - IF the question specifies an industry/process (e.g., "steel", "sintering"), strictly filter the context and exclude irrelevant industries. 
       - IF the question is GENERAL (e.g., "What are VOCs?", "Sources of Methane"), EMBRACE THE DIVERSITY of the context. Categorize your answer by the different industries or applications found in the data (e.g., "In the steel industry...", "In waste management..."). Do not artificially restrict the answer to steel if the prompt is broad.
    4. EXHAUSTIVE ENUMERATION & HIERARCHY: When asked for species or components, EXHAUSTIVELY list ALL valid entities retrieved. Group them logically by chemical family or industry source.
    5. ENTITY TYPE ISOLATION: If asked for 'ControlTech' or technologies, ONLY output equipment/methods (e.g., RTO). DO NOT output chemical species as technologies, and vice versa.
    6. ESCAPE HATCH STRICTLY LIMITED: You are ONLY allowed to output exactly "Not found" if the Context is ABSOLUTELY EMPTY (e.g., []).
    7. IGNORE METADATA: The context may contain database properties like 'source_doc', 'confidence', 'id', or similar technical labels. You MUST completely ignore these in your final output. ONLY synthesize and present the scientific and factual domain knowledge.
    
    8. STRICT CONTEXT MERGING (MULTI-TURN): If the Question contains "--- PREVIOUS CONTEXT ---", you MUST treat the chemical species and data listed there as ABSOLUTE FACTS. You are STRICTLY FORBIDDEN from stating "Specific VOCs were not provided" or "Not specified" if they exist in the PREVIOUS CONTEXT.
    9. ASYMMETRIC MAPPING HANDLING (CRITICAL): When mapping emission sources to control technologies across different processes (e.g., Sintering vs. Coking): IF you have VOC data for a process from the PREVIOUS CONTEXT, but NO ControlTech is found in the current context, you MUST STILL LIST ALL the specific VOCs! Simply state for the technology part: "Control Technology: Not specified in current graph retrieval." DO NOT erase or ignore the VOCs just because the technology is missing.
    10. PRESERVE PROCESS SEPARATION: You must maintain distinct categories for distinct processes (e.g., keep "Sintering" and "Coking" strictly separate) and explicitly list the exact chemical names (e.g., Chloroethane, Benzene, Formaldehyde) under each, mapping them to their specific technologies if available.
    11. EXPLICIT TECH-TO-VOC MAPPING (CRITICAL): When listing a 'Control Technology', you MUST append a brief explanation specifying EXACTLY which VOC species or chemical families (from your listed Key Characteristic VOCs) this technology is designed to mitigate. 
    Example Format: "- [Technology Name] – effective for mitigating [Specific VOCs, e.g., halohydrocarbons, BTX]."
    DO NOT just list the technology without explaining its target pollutants.
    Context: {context}
    Question: {question}

    Professional Answer:"""
    qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])
    kg_rag_chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=llm,
        graph=graph,
        verbose=True,
        cypher_prompt=cypher_prompt, 
        qa_prompt=qa_prompt,  
        allow_dangerous_requests=True,
        top_k=100  
    )
    return kg_rag_chain
# ==========================================
# 4. 🕸️ Core Visualization Function: Generate Pyvis HTML
# ==========================================
def generate_vis_subgraph_html(message, api_key, base_url):
    if not neo4j_driver:
        return "⚠️ Database not connected"

    if not api_key or not api_key.startswith("sk-"):
        return "<h3>⚠️ API Key missing, unable to generate visualization</h3>"

    try:
        llm_kwargs = {"model": "gpt-4o-mini", "temperature": 0, "openai_api_key": api_key}
        if base_url and base_url.strip():
            llm_kwargs["base_url"] = base_url.strip()
        llm = ChatOpenAI(**llm_kwargs)
        
        prompt = f"Please extract the [single most] core entity noun from the following user question (e.g., chemical substances like 'CH4', 'Methane', 'Toluene', or technical equipment like 'vacuum pump', 'PDMS'). Output ONLY this word, and absolutely do NOT include any other punctuation or explanatory content! Question: {message}"
        keyword = llm.invoke(prompt).content.strip()
    except Exception as e:
        keyword = message 
        print(f"Keyword extraction failed: {e}")

    if not keyword:
        keyword = message

    print(f"🔍 Extracted graph keyword for visualization: {keyword}")

    vis_cypher = f"""
    MATCH (core_entity)
    WHERE toLower(core_entity.name) CONTAINS toLower('{keyword}') OR toLower(core_entity.id) CONTAINS toLower('{keyword}') OR toLower(core_entity.canonical_name) CONTAINS toLower('{keyword}')
    MATCH (core_entity)-[r]-(neighbor)
    RETURN core_entity, r, neighbor 
    LIMIT 60
    """
    
    color_map = {
        'ControlTech': '#F7819F', 
        'VOCSpecies': '#81BEF7',  
        'EmissionSource': '#F5D76E', 
        'Process': '#ABEBC6',      
        'Chunk': '#E5E7E9',        
        'Method': '#D7BDE2'        
    }

    nodes = {}
    edges = []

    try:
        with neo4j_driver.session(database=NEO4J_DATABASE_NAME) as session:
            result = session.run(vis_cypher)
            
            for record in result:
                core_node = record['core_entity']
                rel = record['r']
                neighbor_node = record['neighbor']
                
                core_id = core_node.element_id
                if core_id not in nodes:
                    core_label = list(core_node.labels)[0] 
                    core_name = core_node.get('name', core_node.get('id', 'Unknown'))
                    nodes[core_id] = {
                        'id': core_id,
                        'label': core_name,
                        'title': f"Label: {core_label}\n{json.dumps(dict(core_node), indent=2, ensure_ascii=False)}",
                        'color': color_map.get(core_label, '#D2E5FF') 
                    }

                neighbor_id = neighbor_node.element_id
                if neighbor_id not in nodes:
                    neighbor_label = list(neighbor_node.labels)[0]
                    if neighbor_label == 'Chunk':
                         raw_name = neighbor_node.get('text', '')[:12] + '...' 
                    else:
                        raw_name = neighbor_node.get('name', neighbor_node.get('id', 'Unknown'))
                    
                    nodes[neighbor_id] = {
                        'id': neighbor_id,
                        'label': raw_name,
                        'title': f"Label: {neighbor_label}\n{json.dumps(dict(neighbor_node), indent=2, ensure_ascii=False)}",
                        'color': color_map.get(neighbor_label, '#D2E5FF')
                    }

                rel_type = rel.type
                edges.append((rel.start_node.element_id, rel.end_node.element_id, rel_type))

    except Exception as e:
        print(f"❌ Visualization query failed: {str(e)}")
        return f"<h3>❌ Visualization relation query failed: {str(e)}</h3>"

    if not nodes:
        return f"<h3>⚠️ Search complete, but no relational network found for visualization.</h3><p>💡 AI extracted core keyword: <b>[{keyword}]</b>. It's possible this entity is an isolated node in the database, or lacks direct 1st-degree relational nodes.</p>"

    net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='#333333', notebook=False)
    for node_id, node_data in nodes.items():
        net.add_node(node_data['id'], label=node_data['label'], title=node_data['title'], color=node_data['color'])
    for edge in edges:
        net.add_edge(edge[0], edge[1], label=edge[2], width=1, color='#DDDDDD')
    
    net.toggle_physics(True)
    net.set_options("""
    var options = {
      "interaction": {
        "hover": true,
        "hoverConnectedEdges": true
      }
    }
    """)
    
    path = tempfile.mktemp(suffix='.html')
    net.save_graph(path)
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
        
    escaped_html = html.escape(html_content)
    return f'<iframe style="width: 100%; height: 600px; border: none;" srcdoc="{escaped_html}"></iframe>'

# ==========================================
# ==========================================
# 5. Q&A Function (🔥 Dynamic Thinking & Fault-Tolerant Architecture 🔥)
# ==========================================
def answer_question(message, history, api_key, base_url):
    formatted_history = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "AI"
        content = msg["content"]
        if isinstance(content, str):
            if not content.startswith("🧠 [1/3]") and not content.startswith("📖 [2/3]") and not content.startswith("🌐 [3/3]"):
                formatted_history += f"[{role}]: {content}\n\n"
        elif isinstance(content, (list, tuple)):
            formatted_history += f"[{role}]: [多模态或文件内容]\n\n"

    history.append({"role": "user", "content": message})
    
    if not api_key or not api_key.startswith("sk-"):
        history.append({"role": "assistant", "content": "⚠️ Please ensure a valid LLM API Key is entered above."})
        yield history
        return
    
    if graph is None:
        history.append({"role": "assistant", "content": "⚠️ Graph database connection failed, please check environment variables."})
        yield history
        return
    
    llm_kwargs = {"model": "gpt-4o-mini", "temperature": 0, "openai_api_key": api_key}
    if base_url and base_url.strip():
        llm_kwargs["base_url"] = base_url.strip()
    llm = ChatOpenAI(**llm_kwargs)

    if formatted_history.strip():
        enhanced_query = f"--- PREVIOUS CONTEXT ---\n{formatted_history.strip()}\n\n--- CURRENT QUESTION ---\n{message}"
    else:
        enhanced_query = message

    # 💡 Critical Fix: Declare flag variables early to ensure they exist in all scopes
    final_answer = "Not found"
    local_hit = False 
    
    # --- [Tier 1: Graph RAG] ---
    try:
        history.append({"role": "assistant", "content": "🧠 [1/3] Querying Neo4j Knowledge Graph for industry logic..."})
        yield history
        
        chain = get_graph_chain(api_key, base_url) 
        response = chain.invoke({"query": enhanced_query})
        final_answer = response["result"]
    except Exception as e:
        print(f"Graph tier error: {e}")
        final_answer = "Not found"

# --- [Tier 2: Local Vector Database with VISIBLE THINKING] ---
    if final_answer.strip().strip('"').strip("'") == "Not found":
        
        history[-1]["content"] = f"📖 [2/3] Graph missed. Utilizing high-dimensional semantic search on the local vector database..."
        yield history
        
        try:
            if vectorstore:
                # Use the original sentence for semantic retrieval to preserve the complete intent
                docs = vectorstore.similarity_search(message, k=8)
                
                if docs:
                    history[-1]["content"] += f"\n\n*🔍 Retrieved {len(docs)} local documents, evaluating relevance...*"
                    yield history
                    time.sleep(1) # Pause to simulate processing time
                    
                    local_context = "\n\n".join([f"[{d.metadata.get('title', 'Unknown')}] {d.page_content[:800]}" for d in docs])
                    
                    judge_prompt = f"""You are an authoritative environmental engineering expert. 
Your task is to answer the user's question '{message}' based ONLY on the provided documents.

[CRITICAL RULES]:
1. ENTITY TYPE ISOLATION (CRITICAL): Strictly match the requested entity type. If the user asks for 'ControlTech' or technologies, ONLY output equipment or treatment methods (e.g., RTO, Scrubbers, Activated Carbon). DO NOT output chemical species as technologies.
2. ADAPTIVE SCOPE FOCUS: 
   - IF the user's question explicitly mentions a specific industry (e.g., "steel"), filter out irrelevant industries.
   - IF the question is GENERAL (e.g., "Which sources release Methane?", "Common VOC control technologies"), you MUST synthesize a broad answer encompassing ALL relevant industries found in the documents (e.g., Landfills, Natural Gas, Steel) and categorize them clearly.
3. DATA SYNTHESIS: Provide a summarized, professional answer based on what IS available. Group species, sources, or technologies logically with bullet points.
4. ESCAPE HATCH: ONLY if the documents have absolutely ZERO relation to the specific entity type requested, you MUST reply exactly with: 'PASS'

Documents:
{local_context}

Professional Answer:"""
                    
                    local_answer = llm.invoke(judge_prompt).content.strip()

                    if local_answer.upper() != "PASS" and not local_answer.upper().startswith("PASS"):
                        final_answer = f"**📚 [Local DB] Answer formulated based on domain-specific literature:**\n\n{local_answer}"
                        local_hit = True
                    else:
                        history[-1]["content"] += "\n\n*⚠️ Local documents evaluated as completely irrelevant by the LLM (PASS triggered). Insufficient context.*"
                        yield history
                        time.sleep(1.5)
                else:
                    history[-1]["content"] += "\n\n*⚠️ No relevant segments retrieved from the local database.*"
                    yield history
                    time.sleep(1)
        except Exception as e:
            print(f"Vector tier error: {e}")
    # --- [Tier 3: Wikipedia] ---
    if not local_hit and final_answer.strip().strip('"').strip("'") == "Not found":
        history[-1]["content"] += "\n\n🌐 [3/3] Calling Wikipedia API for global knowledge fallback..."
        yield history
        
        try:
            wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
            wiki_kw = llm.invoke(f"Extract a core noun from the following sentence: {message}").content.strip()
            wiki_res = wiki_tool.run(wiki_kw)
            
            if "No good" not in wiki_res:
                summary = llm.invoke(f"Summarize the following to answer '{message}': {wiki_res}").content
                final_answer = f"**🌐 [Web API] Global knowledge fallback:**\n\n{summary}"
            else:
                final_answer = "😔 Sorry, no relevant information was found across all three retrieval tiers."
        except Exception as e:
            final_answer = f"System Error: {e}"

    # Final Streaming Output
    streamed_text = ""
    for char in final_answer:
        streamed_text += char
        history[-1]["content"] = streamed_text 
        yield history
        time.sleep(0.005) 

# ==========================================
# 6. UI Construction (Professional English Theme)
# ==========================================
custom_theme = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    text_size=gr.themes.sizes.text_md
)

with gr.Blocks() as demo:
    gr.Markdown("# 🌿 Chat-ISV: Intelligent Q&A System for VOCs in the Iron and Steel Industry")
    
    gr.Markdown("""**💡 Recommended Question Examples:**
    1. Which EmissionSources release Methane (CH4)?
    2. List common VOCs ControlTech, excluding laboratory equipment.
    3. What is the basic concept of Volatile Organic Compounds (VOCs)?
    4. What Methods in the graph can be used to observe VOCsSpecies?
    5. What specific VOC species are emitted from the iron and steel industry?""")
    
    with gr.Row():
        api_key_input = gr.Textbox(
            label="🔑 OpenAI API Key",
            placeholder="sk-...",
            type="password",
            scale=1
        )
        base_url_input = gr.Textbox(
            label="🌐 Base URL (Optional)",
            placeholder="",
            value=DEFAULT_BASE_URL,
            scale=1
        )
    
    gr.Markdown("---")
    
    gr.Markdown("### 💬 Chat-ISV: Domain Knowledge LLM Assistant")
    
    chatbot_ui = gr.Chatbot(
        avatar_images=(
            "user.jpg",   
            "ai.jpg"      
        ),
        height=400,
        render=True 
    )
    
    msg_input = gr.Textbox(
        label="💬 Enter your professional question here and press Enter",
        placeholder="e.g.: Which emission sources release toluene?",
        lines=1,
        max_lines=3
    )
    
    with gr.Row():
        submit_btn = gr.Button("🚀 Submit Question", variant="primary")
        clear_btn = gr.Button("🗑️ Clear Chat")

    gr.Markdown("---")
    
    gr.Markdown("### 🕸️ Dynamic Subgraph Network Navigation")
    gr.Markdown("*(Note: Upon answering, this area generates a 1st-degree relational subgraph centered on the entities in your question. Supports zoom, drag, and highlight.)*")
    
    html_vis_output = gr.HTML(
        value="<h3>💬 Please enter credentials above and send a question...</h3><p>The LLM will translate your question into query code and present a professional dynamic subgraph visualization here.</p>",
        height="600px", 
        show_label=True,
        render=True
    )
    
    gr.Markdown("---")
    gr.Markdown("### 🕸️ Graph Global Schema Details")
    
    schema_output = gr.Textbox(
        label="Current Graph Database Schema and Relational Distribution - Click to fetch Schema", 
        lines=10, 
        max_lines=20
    )
    def get_schema():
        return graph.schema if graph else "⚠️ Database not connected"
    
    refresh_btn = gr.Button("🔄 Refresh Graph Schema")
    refresh_btn.click(fn=get_schema, outputs=schema_output)

    # ==========================================
    # Wiring logic
    # ==========================================
    answer_event = msg_input.submit(
        fn=answer_question,
        inputs=[msg_input, chatbot_ui, api_key_input, base_url_input],
        outputs=[chatbot_ui]
    )
    answer_event.then(
        fn=generate_vis_subgraph_html,
        inputs=[msg_input, api_key_input, base_url_input],
        outputs=[html_vis_output]
    )
    answer_event.then(lambda: "", outputs=[msg_input])

    btn_event = submit_btn.click(
        fn=answer_question,
        inputs=[msg_input, chatbot_ui, api_key_input, base_url_input],
        outputs=[chatbot_ui]
    )
    btn_event.then(
        fn=generate_vis_subgraph_html,
        inputs=[msg_input, api_key_input, base_url_input],
        outputs=[html_vis_output]
    )
    btn_event.then(lambda: "", outputs=[msg_input])

    clear_btn.click(lambda: [], outputs=chatbot_ui, queue=False)
    clear_btn.click(lambda: "<h3>💬 Please send a question...</h3>", outputs=html_vis_output)

if __name__ == "__main__":
    demo.launch(
        theme=custom_theme,
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
