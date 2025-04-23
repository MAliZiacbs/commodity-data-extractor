# app.py

import streamlit as st
import pandas as pd
import json
import requests
from document_processor import DocumentProcessor

# Set up page configuration
st.set_page_config(
    page_title="Commodity Data Extractor",
    page_icon="📊",
    layout="wide"
)

# Access API configuration from Streamlit secrets
API_URL = st.secrets.get("API_URL", "https://adb-360063509637705.5.azuredatabricks.net/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations")
API_TOKEN = st.secrets.get("API_TOKEN", "")

def extract_commodity_data(document_content):
    """Extract structured information from the document using Llama 3"""
    
    # Construct the system prompt for Llama 3
    prompt = """<|system|>
You are an expert data extraction system specialized in analyzing commodity strategy documents.
Your task is to extract specific information from documents and structure it as a valid JSON.
Focus only on extracting factual information present in the document.
When information is missing, use null or empty arrays rather than making up information.
</|system|>

<|user|>
Please analyze the following commodity strategy document and extract this information into a JSON structure:

1. commodity_name: The name of the commodity being discussed (e.g., Sugar, Dairy, Oils)
2. responsible_managers: Who is responsible for this commodity
3. creation_date: When the document was created
4. valid_until: The expiration date of the strategy
5. cost_drivers: A dictionary containing cost breakdown components (like labor, raw materials, energy) with their percentages
6. quantitative_initiatives: An array of initiatives with their IDs, descriptions, values in EUR, and status
7. qualitative_initiatives: An array of non-monetary initiatives 
8. swot_analysis: A dictionary with arrays for strengths, weaknesses, opportunities, and threats
9. sustainability_factors: Any sustainability information like deforestation risk, emissions, etc.

Document content:
{content}

Return ONLY a valid JSON object with no additional text. If information is not available, include the key with an empty value or appropriate placeholder.
</|user|>"""
    
    # Replace placeholder with actual content
    prompt = prompt.replace("{content}", document_content)
    
    # Prepare the request payload for Databricks serving endpoint
    payload = {
        "inputs": [{
            "prompt": prompt,
            "temperature": 0.1,
            "max_tokens": 4000
        }]
    }
    
    # Set up headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    
    try:
        # Make the API call
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            
            # Extract the assistant's response
            if "predictions" in result and len(result["predictions"]) > 0:
                response_text = result["predictions"][0]
                
                # Parse JSON from the response
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].strip()
                else:
                    # Find the first opening brace and the last closing brace
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = response_text[start:end]
                    else:
                        json_str = response_text
                
                # Parse the JSON
                try:
                    data = json.loads(json_str)
                    return data
                except json.JSONDecodeError as e:
                    return {"error": f"Failed to parse JSON: {str(e)}", "raw_response": response_text}
            else:
                return {"error": "No predictions in response", "raw_response": result}
        elif response.status_code == 400:
            # Check if this is a token limit issue
            if "context_length_exceeded" in str(response.text) or "maximum context length" in str(response.text):
                return {
                    "error": "Document too large for model's context window",
                    "solution": "Try extracting specific information or processing in chunks"
                }
            else:
                return {"error": f"API Error (400): {response.text}"}
        elif response.status_code == 401 or response.status_code == 403:
            return {"error": "Authentication error - check your API token"}
        elif response.status_code >= 500:
            return {"error": "Server error - the API service may be experiencing issues"}
        else:
            return {"error": f"API Error ({response.status_code}): {response.text}"}
            
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def process_large_document(document_content, max_tokens=7000):
    """Split large documents into processable chunks"""
    
    # Simple chunking strategy - split by pages/slides
    chunks = []
    current_chunk = ""
    current_token_count = 0
    estimated_tokens_per_word = 1.3
    
    # Split by page/slide markers or just by paragraphs if no markers
    if "--- Page" in document_content or "--- Slide" in document_content:
        # Split by page/slide markers
        sections = []
        for line in document_content.split("\n"):
            if line.startswith("--- Page") or line.startswith("--- Slide"):
                sections.append(line)
            elif sections:
                sections[-1] += "\n" + line
            else:
                sections.append(line)
    else:
        # Split by paragraphs (double newlines)
        sections = document_content.split("\n\n")
    
    # Group sections into chunks
    for section in sections:
        section_tokens = len(section.split()) * estimated_tokens_per_word
        
        # If adding this section exceeds our limit, create a new chunk
        if current_token_count + section_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = section
            current_token_count = section_tokens
        else:
            # Otherwise add to current chunk
            if current_chunk:
                current_chunk += "\n\n" + section
            else:
                current_chunk = section
            current_token_count += section_tokens
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def extract_from_large_document(document_content):
    """Process a large document by extracting from chunks and merging results"""
    # Check if document needs chunking
    token_estimate = len(document_content.split()) * 1.3
    
    if token_estimate > 7000:  # Less than model's max to allow for prompt and response
        st.warning(f"Document is large ({int(token_estimate)} estimated tokens). Processing in multiple steps...")
        
        # Split document into chunks
        chunks = process_large_document(document_content)
        st.info(f"Document split into {len(chunks)} chunks for processing")
        
        # Process each chunk
        all_results = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Processing chunk {i+1} of {len(chunks)}..."):
                result = extract_commodity_data(chunk)
                if "error" not in result:
                    all_results.append(result)
                else:
                    st.warning(f"Issue processing chunk {i+1}: {result.get('error')}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(chunks))
        
        # Merge results
        if all_results:
            merged_result = {}
            
            # Take basic info from the first successful result
            first_result = all_results[0]
            for field in ["commodity_name", "responsible_managers", "creation_date", "valid_until"]:
                if field in first_result:
                    merged_result[field] = first_result[field]
            
            # Merge cost drivers (take the most complete one)
            if any("cost_drivers" in result for result in all_results):
                most_complete = max(
                    [result.get("cost_drivers", {}) for result in all_results if "cost_drivers" in result],
                    key=lambda x: len(x) if isinstance(x, dict) else 0,
                    default={}
                )
                merged_result["cost_drivers"] = most_complete
            
            # Merge arrays from all results
            for field in ["quantitative_initiatives", "qualitative_initiatives"]:
                all_items = []
                for result in all_results:
                    if field in result and isinstance(result[field], list):
                        all_items.extend(result[field])
                
                # Remove duplicates based on descriptions
                if all_items:
                    unique_items = []
                    descriptions = set()
                    for item in all_items:
                        if isinstance(item, dict) and "description" in item:
                            desc = item["description"]
                            if desc not in descriptions:
                                descriptions.add(desc)
                                unique_items.append(item)
                        else:
                            unique_items.append(item)
                    merged_result[field] = unique_items
                else:
                    merged_result[field] = []
            
            # Merge SWOT
            swot_merged = {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}
            for result in all_results:
                if "swot_analysis" in result and isinstance(result["swot_analysis"], dict):
                    for key in ["strengths", "weaknesses", "opportunities", "threats"]:
                        if key in result["swot_analysis"] and isinstance(result["swot_analysis"][key], list):
                            swot_merged[key].extend(result["swot_analysis"][key])
            
            # Remove duplicates in SWOT
            for key in swot_merged:
                swot_merged[key] = list(set(swot_merged[key]))
            
            merged_result["swot_analysis"] = swot_merged
            
            # Merge sustainability factors (take the most complete one)
            if any("sustainability_factors" in result for result in all_results):
                most_complete = max(
                    [result.get("sustainability_factors", {}) for result in all_results if "sustainability_factors" in result],
                    key=lambda x: len(x) if isinstance(x, dict) else 0,
                    default={}
                )
                merged_result["sustainability_factors"] = most_complete
            
            return merged_result
        else:
            return {"error": "Failed to process any chunks successfully"}
    else:
        # Document is small enough to process in one go
        return extract_commodity_data(document_content)

def display_structured_data(data):
    """Display the extracted data in a structured format"""
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Overview", "SWOT & Initiatives", "Raw JSON"])
    
    with tab1:
        st.header("Commodity Overview")
        
        # Basic info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Commodity", data.get("commodity_name", "Unknown"))
            st.write(f"**Created:** {data.get('creation_date', 'Unknown')}")
        with col2:
            st.write(f"**Manager:** {data.get('responsible_managers', 'Unknown')}")
            st.write(f"**Valid until:** {data.get('valid_until', 'Unknown')}")
        
        # Cost drivers
        st.subheader("Cost Drivers")
        cost_drivers = data.get("cost_drivers", {})
        if cost_drivers and isinstance(cost_drivers, dict):
            # Clean percentage values for chart
            cleaned_data = {}
            for k, v in cost_drivers.items():
                if isinstance(v, str):
                    # Handle percentage strings
                    cleaned_val = v.replace('%', '').replace('k€', '').strip()
                    try:
                        cleaned_data[k] = float(cleaned_val)
                    except ValueError:
                        # If conversion fails, use original
                        cleaned_data[k] = v
                else:
                    cleaned_data[k] = v
            
            # Create a horizontal bar chart
            cost_df = pd.DataFrame({
                'Component': list(cleaned_data.keys()),
                'Value': list(cleaned_data.values())
            })
            st.bar_chart(cost_df.set_index('Component'))
        else:
            st.write("No cost driver information available")
    
    with tab2:
        # SWOT Analysis
        st.header("SWOT Analysis")
        swot = data.get("swot_analysis", {})
        if swot and isinstance(swot, dict):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Strengths")
                strengths = swot.get("strengths", [])
                for s in strengths:
                    st.markdown(f"✅ {s}")
                
                st.subheader("Opportunities")
                opportunities = swot.get("opportunities", [])
                for o in opportunities:
                    st.markdown(f"🚀 {o}")
            
            with col2:
                st.subheader("Weaknesses")
                weaknesses = swot.get("weaknesses", [])
                for w in weaknesses:
                    st.markdown(f"⚠️ {w}")
                
                st.subheader("Threats")
                threats = swot.get("threats", [])
                for t in threats:
                    st.markdown(f"❌ {t}")
        else:
            st.write("No SWOT analysis data available")
        
        # Initiatives
        st.header("Initiatives")
        
        # Quantitative initiatives
        quant_initiatives = data.get("quantitative_initiatives", [])
        if quant_initiatives and isinstance(quant_initiatives, list) and len(quant_initiatives) > 0:
            st.subheader("Quantitative Initiatives")
            
            # Convert to DataFrame for better display
            init_df = pd.DataFrame(quant_initiatives)
            st.dataframe(init_df)
        else:
            st.write("No quantitative initiatives available")
        
        # Qualitative initiatives
        qual_initiatives = data.get("qualitative_initiatives", [])
        if qual_initiatives and isinstance(qual_initiatives, list) and len(qual_initiatives) > 0:
            st.subheader("Qualitative Initiatives")
            
            # Convert to DataFrame for better display
            qual_df = pd.DataFrame(qual_initiatives)
            st.dataframe(qual_df)
        else:
            st.write("No qualitative initiatives available")
        
        # Sustainability
        st.subheader("Sustainability Factors")
        sustainability = data.get("sustainability_factors", {})
        if sustainability and isinstance(sustainability, dict):
            for key, value in sustainability.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.write("No sustainability information available")
    
    with tab3:
        # Raw JSON view
        st.header("Raw JSON Data")
        st.json(data)
        
        # Download button
        json_str = json.dumps(data, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"{data.get('commodity_name', 'commodity')}_data.json",
            mime="application/json"
        )

# App title and description
st.title("Commodity Strategy Data Extractor")
st.write("Upload a commodity strategy document (PDF or PowerPoint) to extract structured information.")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.write("""
    This application uses Meta's Llama 3 70B model to extract structured data 
    from commodity strategy documents.
    
    Upload your document to begin the extraction process.
    """)
    
    # Show environment info (for debugging, can be removed in production)
    if st.checkbox("Show API Configuration", False):
        st.write(f"API URL: {API_URL}")
        st.write(f"API Token set: {'Yes' if API_TOKEN else 'No'}")

# Main content - File upload section
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "pptx", "ppt"])

if uploaded_file is not None:
    st.write(f"Uploaded: **{uploaded_file.name}**")
    
    # Extract text based on file type
    with st.spinner("Processing document..."):
        try:
            document_content, file_type = DocumentProcessor.process_document(uploaded_file)
            st.success(f"{file_type} processed successfully")
            
            # Show sample of extracted text
            with st.expander("Preview extracted text"):
                preview_length = min(2000, len(document_content))
                st.text_area("Document Content", document_content[:preview_length] + ("..." if len(document_content) > preview_length else ""), height=200)
                
                # Show token count estimate (rough approximation)
                token_estimate = len(document_content.split()) * 1.3  # Rough estimate
                st.caption(f"Estimated token count: {int(token_estimate)}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.stop()
    
    # Extract data button
    if st.button("Extract Structured Data"):
        with st.spinner("Analyzing with Llama 3 70B... (this may take a minute)"):
            # Call function that handles large documents
            result = extract_from_large_document(document_content)
            
            if "error" in result:
                st.error(f"Error extracting data: {result['error']}")
                
                # Show raw response if available (for debugging)
                if "raw_response" in result:
                    with st.expander("Raw response"):
                        st.text(result["raw_response"])
                
                # Show solution if available
                if "solution" in result:
                    st.info(f"Suggestion: {result['solution']}")
            else:
                # Display the structured data
                display_structured_data(result)

# Footer
st.markdown("---")
st.caption("Powered by Meta Llama 3 70B")