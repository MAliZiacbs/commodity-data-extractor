# api_client.py

import json
import requests
import os
from typing import Dict, Any, Optional

class LlamaApiClient:
    """Client for interacting with the Databricks Llama 3 API endpoint"""
    
    def __init__(self):
        self.api_url = os.environ.get("API_URL", "https://adb-360063509637705.5.azuredatabricks.net/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations")
        self.api_token = os.environ.get("API_TOKEN", "")
        
        if not self.api_token:
            print("Warning: API_TOKEN environment variable not set")
    
    def extract_commodity_data(self, document_content: str) -> Dict[str, Any]:
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
                "temperature": 0.1,  # Lower temperature for more deterministic outputs
                "max_tokens": 4000   # Ensure enough tokens for the response
            }]
        }
        
        # Set up headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }
        
        try:
            # Make the API call
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120  # Longer timeout as Llama 3 70B can take time
            )
            
            # Check for successful response
            if response.status_code == 200:
                result = response.json()
                
                # Extract the assistant's response
                if "predictions" in result and len(result["predictions"]) > 0:
                    response_text = result["predictions"][0]
                    
                    # Parse JSON from the response
                    # Sometimes the model might include markdown formatting
                    if "```json" in response_text:
                        json_str = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        json_str = response_text.split("```")[1].strip()
                    else:
                        # Remove any non-JSON text the model might add
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
            else:
                return {"error": f"API Error: {response.status_code}", "details": response.text}
                
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}