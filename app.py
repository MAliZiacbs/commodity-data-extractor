# app.py

import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from api_client import LlamaApiClient
from document_processor import DocumentProcessor

# Load environment variables
load_dotenv()

# Initialize API client
api_client = LlamaApiClient()

# Set up page configuration
st.set_page_config(
    page_title="Commodity Data Extractor",
    page_icon="📊",
    layout="wide"
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
        st.write(f"API URL: {api_client.api_url}")
        st.write(f"API Token set: {'Yes' if api_client.api_token else 'No'}")

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
        with st.spinner("Analyzing with Llama 3 70B... (this may take up to a minute)"):
            # Call API to extract data
            result = api_client.extract_commodity_data(document_content)
            
            if "error" in result:
                st.error(f"Error extracting data: {result['error']}")
                
                # Show raw response if available (for debugging)
                if "raw_response" in result:
                    with st.expander("Raw response"):
                        st.text(result["raw_response"])
            else:
                # Display the structured data
                display_structured_data(result)

# Footer
st.markdown("---")
st.caption("Powered by Meta Llama 3 70B")