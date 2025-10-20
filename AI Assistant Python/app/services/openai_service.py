import httpx
import json
import logging
from typing import Optional, List
import os
from ..models.schemas import IncidentAnalysis, TrainingDataResponse, KnowledgeBaseResponse
from ..models.database import TrainingData, KnowledgeBase

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        logger.info(f"Azure OpenAI Configuration - API Key: {'***' + self.api_key[-4:] if self.api_key else 'None'}")
        logger.info(f"Azure OpenAI Configuration - Endpoint: {self.endpoint}")
        logger.info(f"Azure OpenAI Configuration - Deployment ID: {self.deployment_id}")
        logger.info(f"Azure OpenAI Configuration - API Version: {self.api_version}")
        
        if not self.api_key or "PUT-YOUR-" in self.api_key or not self.endpoint or not self.deployment_id:
            logger.warning("Azure OpenAI configuration incomplete. AI analysis will use fallback mode.")
            logger.warning(f"Missing: API Key={not bool(self.api_key)}, Endpoint={not bool(self.endpoint)}, Deployment={not bool(self.deployment_id)}")
            self.configured = False
        else:
            logger.info("Azure OpenAI configuration complete. AI analysis enabled.")
            self.configured = True
    
    # --- START of the new function ---
    async def is_valid_incident_async(self, description: str) -> bool:
        """
        Uses AI to quickly classify if a description is a valid incident or not.
        """
        if not self.configured:
            logger.warning("Using fallback validation - Azure OpenAI not configured properly")
            # Basic keyword check as a fallback
            return len(description.split()) > 2

        try:
            prompt = f"""
            You are a validation bot. Your task is to classify the following text as either a "valid incident" or an "invalid prompt".
            - A "valid incident" is a description of a technical or operational problem, like 'Vessel ETA is not updated' or 'Container information is duplicated'.
            - An "invalid prompt" is a short, generic, or nonsensical input, like 'yes', 'hello', 'asdfghjkl', or a question that is not an incident description.

            Text to classify: "{description}"

            Classification (valid incident or invalid prompt):
            """

            request_body = {
                "messages": [
                    {"role": "system", "content": "You are a validation bot."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 10,
                "temperature": 0.0
            }

            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            azure_url = f"{self.endpoint}/openai/deployments/{self.deployment_id}/chat/completions?api-version={self.api_version}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(azure_url, json=request_body, headers=headers, timeout=15.0)

                if response.is_success:
                    response_data = response.json()
                    classification = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
                    return "valid incident" in classification
                else:
                    logger.error(f"Validation API error: {response.status_code} - {response.text}")
                    # Fallback to true to avoid blocking the user if the validation service fails
                    return True
        except Exception as ex:
            logger.error(f"Error calling validation API: {ex}")
            # Fallback to true to avoid blocking the user
            return True
    # --- END of the new function ---
    
    async def analyze_image_async(self, image_base64: str, incident_description: str = "") -> str:
        """Analyze image using Azure OpenAI Vision API"""
        if not self.configured:
            logger.warning("Using fallback image analysis - Azure OpenAI not configured")
            return "[Image Analysis: Visual analysis shows maritime incident documentation. Azure OpenAI Vision would extract detailed incident information including equipment damage, operational issues, safety concerns, and environmental conditions visible in the image.]"
        
        try:
            prompt = f"""You are an expert maritime operations analyst. Analyze this incident image and provide detailed observations.

            Context: {incident_description if incident_description else 'Maritime incident analysis'}
            
            Please identify and describe:
            1. Equipment or infrastructure visible
            2. Any visible damage, issues, or anomalies
            3. Safety concerns or hazards
            4. Environmental conditions
            5. Personnel or operational context
            6. Specific maritime/port operations details
            
            Provide a concise but detailed analysis focusing on incident-relevant details."""

            request_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Use Azure OpenAI endpoint with vision capabilities
            azure_url = f"{self.endpoint}/openai/deployments/{self.deployment_id}/chat/completions?api-version={self.api_version}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(azure_url, json=request_body, headers=headers, timeout=30.0)
                
                if response.is_success:
                    response_data = response.json()
                    vision_analysis = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    logger.info(f"Vision analysis completed successfully")
                    return vision_analysis
                else:
                    logger.error(f"Vision API error: {response.status_code} - {response.text}")
                    return "[Image analysis failed - API error]"
                    
        except Exception as ex:
            logger.error(f"Error in vision analysis: {ex}")
            return "[Image analysis encountered an error]"

    async def analyze_incident_async(self, description: str, training_data: List[TrainingData] = None, knowledge_data: List[KnowledgeBase] = None) -> IncidentAnalysis:
        """Analyze incident using AI with training data and knowledge base context"""
        
        # If configuration incomplete, use fallback
        if not self.configured:
            logger.warning("Using fallback analysis - Azure OpenAI not configured properly")
            return self._create_fallback_analysis(description, training_data, knowledge_data)
        
        logger.info("Using Azure OpenAI for incident analysis")
        
        try:
            prompt = self._create_analysis_prompt(description, training_data, knowledge_data)
            
            request_body = {
                "messages": [
                    {"role": "system", "content": "You are an expert maritime operations analyst for PORTNET®."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 800,
                "temperature": 0.3
            }
            
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Use Azure OpenAI endpoint
            azure_url = f"{self.endpoint}/openai/deployments/{self.deployment_id}/chat/completions?api-version={self.api_version}"
            logger.info(f"Making request to Azure OpenAI: {azure_url}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(azure_url, json=request_body, headers=headers, timeout=30.0)
                
                if response.is_success:
                    response_data = response.json()
                    ai_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    logger.info(f"OpenAI Response: {ai_content}")
                    return self._parse_analysis_response(ai_content)
                else:
                    error_content = response.text
                    logger.error(f"OpenAI API error: {response.status_code} - {error_content}")
                    return self._create_fallback_analysis(description)
                    
        except Exception as ex:
            logger.error(f"Error calling OpenAI API: {ex}")
            return self._create_fallback_analysis(description)
    
    def _create_analysis_prompt(self, description: str, training_examples: List[TrainingData] = None, knowledge_entries: List[KnowledgeBase] = None) -> str:
        """Create analysis prompt with training data and knowledge context"""
        
        training_section = ""
        if training_examples:
            training_section = "\nTRAINING EXAMPLES (Use these as reference for similar incidents):\n"
            for i, example in enumerate(training_examples):
                training_section += f"""
Example {i + 1}:
Description: {example.incident_description}
Type: {example.expected_incident_type}
Pattern: {example.expected_pattern_match}
Root Cause: {example.expected_root_cause}
Impact: {example.expected_impact}
Urgency: {example.expected_urgency}
Affected Systems: {', '.join(example.expected_affected_systems)}
---"""
        
        knowledge_section = ""
        if knowledge_entries:
            knowledge_section = "\nKNOWLEDGE BASE (Use this information to enhance your analysis):\n"
            for i, entry in enumerate(knowledge_entries):
                knowledge_section += f"""
Knowledge {i + 1} - {entry.title} ({entry.type}):
{entry.content[:500]}{'...' if len(entry.content) > 500 else ''}
Category: {entry.category}
Keywords: {entry.keywords}
---"""
        
        prompt = f"""Analyze this maritime/port operations incident and provide a structured analysis:

INCIDENT DESCRIPTION: {description}
{training_section}
{knowledge_section}

Please provide your analysis in the following JSON format:
{{
    "incident_type": "Brief categorization (e.g., Container Management, Vessel Operations, EDI Processing, etc.)",
    "pattern_match": "What pattern or category this incident matches",
    "root_cause": "Likely root cause based on the description and knowledge base",
    "impact": "Potential impact on operations",
    "urgency": "Low/Medium/High/Critical based on operational impact",
    "affected_systems": ["List of systems that might be affected"]
}}

Focus on maritime operations context including PORTNET®, container management, vessel operations, EDI messaging, terminal operations, and billing systems."""
        
        return prompt
    
    def _parse_analysis_response(self, ai_response: str) -> IncidentAnalysis:
        """Parse AI response into IncidentAnalysis object"""
        try:
            # Try to extract JSON from response
            start = ai_response.find('{')
            end = ai_response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = ai_response[start:end]
                data = json.loads(json_str)
                
                return IncidentAnalysis(
                    incident_type=data.get("incident_type", "System Issue"),
                    pattern_match=data.get("pattern_match", "General incident"),
                    root_cause=data.get("root_cause", "Under investigation"),
                    impact=data.get("impact", "Operational impact being assessed"),
                    urgency=data.get("urgency", "Medium"),
                    affected_systems=data.get("affected_systems", [])
                )
        except Exception as ex:
            logger.error(f"Error parsing AI response: {ex}")
        
        # Fallback parsing if JSON fails
        return self._create_fallback_analysis_from_text(ai_response)
    
    def _create_fallback_analysis(self, description: str, training_data=None, knowledge_data=None) -> IncidentAnalysis:
        """Create fallback analysis when AI is not available, enhanced with database knowledge"""
        description_lower = description.lower()
        
        # Simple pattern matching
        incident_type = "System Issue"
        affected_systems = []
        urgency = "Medium"
        
        # Pattern matching logic
        if any(word in description_lower for word in ["container", "cmau", "gesu", "trlu"]):
            incident_type = "Container Management"
            affected_systems = ["Container Management System", "PORTNET®"]
        elif any(word in description_lower for word in ["vessel", "ship", "mv "]):
            incident_type = "Vessel Operations"
            affected_systems = ["Vessel Management System", "PORTNET®"]
            urgency = "High"
        elif any(word in description_lower for word in ["edi", "message", "ref-ift"]):
            incident_type = "EDI Processing"
            affected_systems = ["EDI System", "Message Processing"]
        elif any(word in description_lower for word in ["gate", "truck", "access"]):
            incident_type = "Terminal Operations"
            affected_systems = ["Gate System", "Access Control"]
        elif any(word in description_lower for word in ["billing", "invoice", "charge"]):
            incident_type = "Financial Operations"
            affected_systems = ["Billing System", "Financial Module"]
        
        # Urgency assessment
        if any(word in description_lower for word in ["critical", "urgent", "error", "failure", "stuck"]):
            urgency = "High"
        elif any(word in description_lower for word in ["minor", "cosmetic"]):
            urgency = "Low"
        
        # Generate root cause analysis using database knowledge
        root_cause = self._generate_database_root_cause(description, training_data, knowledge_data)
        
        return IncidentAnalysis(
            incident_type=incident_type,
            pattern_match=f"Rule-based match: {incident_type}",
            root_cause=root_cause,
            impact="Operational impact being assessed through system analysis",
            urgency=urgency,
            affected_systems=affected_systems
        )
    
    def _generate_database_root_cause(self, description: str, training_data=None, knowledge_data=None) -> str:
        """Generate root cause analysis using database knowledge"""
        description_lower = description.lower()
        
        # If we have training data, look for the most relevant solutions
        if training_data and len(training_data) > 0:
            # Find the most relevant training examples
            best_matches = []
            for data in training_data[:5]:  # Take top 5 matches
                if hasattr(data, 'solution') and data.solution:
                    solution_text = str(data.solution).strip()
                    if solution_text and solution_text != "None":
                        best_matches.append({
                            'problem': getattr(data, 'problem_statement', 'Unknown issue'),
                            'solution': solution_text,
                            'usefulness': int(getattr(data, 'usefulness_count', 0) or 0)
                        })
            
            if best_matches:
                # Sort by usefulness
                best_matches.sort(key=lambda x: x['usefulness'], reverse=True)
                top_solution = best_matches[0]
                
                # Generate specific root cause based on the best matching solution
                root_cause = f"Based on similar incidents in the database: {top_solution['solution']}"
                
                # Add additional context from knowledge base if available
                if knowledge_data and len(knowledge_data) > 0:
                    kb_context = []
                    for kb in knowledge_data[:2]:  # Take top 2 knowledge entries
                        if hasattr(kb, 'solution') and kb.solution:
                            kb_solution = str(kb.solution).strip()
                            if kb_solution and kb_solution != "None":
                                kb_context.append(kb_solution)
                    
                    if kb_context:
                        root_cause += f" Additional guidance: {' '.join(kb_context[:200])}..."  # Limit length
                
                return root_cause
        
        # If we have knowledge base data but no training data matches
        if knowledge_data and len(knowledge_data) > 0:
            kb_solutions = []
            for kb in knowledge_data[:3]:  # Take top 3 knowledge entries
                if hasattr(kb, 'solution') and kb.solution:
                    kb_solution = str(kb.solution).strip()
                    if kb_solution and kb_solution != "None":
                        kb_solutions.append(kb_solution)
            
            if kb_solutions:
                return f"Based on knowledge base guidance: {' '.join(kb_solutions[:300])}..."  # Limit length
        
        # Fallback to pattern-based analysis if no database knowledge available
        if "container" in description_lower and any(word in description_lower for word in ["stuck", "error", "failure"]):
            return "Container processing workflow interrupted. Likely causes: EDI message corruption, database lock, or system timeout during container status update."
        elif "vessel" in description_lower and "arrival" in description_lower:
            return "Vessel arrival processing issue. Possible causes: Port schedule conflict, berth availability problem, or EDI message validation failure."
        elif "edi" in description_lower and "message" in description_lower:
            return "EDI message processing failure. Common causes: Invalid message format, missing required fields, or communication timeout with external systems."
        elif "gate" in description_lower:
            return "Terminal gate operation disruption. Potential causes: Access control system malfunction, container verification failure, or database connectivity issue."
        elif "billing" in description_lower:
            return "Financial transaction processing error. Likely causes: Rate calculation error, missing charge configuration, or invoice generation failure."
        elif any(word in description_lower for word in ["timeout", "slow", "performance"]):
            return "System performance degradation. Possible causes: Database query optimization needed, high server load, or network latency issues."
        else:
            return "Incident requires detailed analysis. Check system logs, verify database connectivity, and review recent system changes for potential root causes."
    
    def _create_fallback_analysis_from_text(self, ai_response: str) -> IncidentAnalysis:
        """Create analysis from AI text response when JSON parsing fails"""
        # Extract information from text response
        lines = ai_response.split('\n')
        
        incident_type = "System Issue"
        root_cause = "Under investigation"
        urgency = "Medium"
        affected_systems = []
        
        for line in lines:
            line_lower = line.lower()
            if "type:" in line_lower or "category:" in line_lower:
                incident_type = line.split(":", 1)[1].strip() if ":" in line else incident_type
            elif "cause:" in line_lower:
                root_cause = line.split(":", 1)[1].strip() if ":" in line else root_cause
            elif "urgency:" in line_lower or "priority:" in line_lower:
                urgency = line.split(":", 1)[1].strip() if ":" in line else urgency
            elif "systems:" in line_lower:
                systems_text = line.split(":", 1)[1].strip() if ":" in line else ""
                if systems_text:
                    affected_systems = [s.strip() for s in systems_text.split(",")]
        
        return IncidentAnalysis(
            incident_type=incident_type[:100] if incident_type else "System Issue",
            pattern_match="AI analysis (text format)",
            root_cause=root_cause[:500] if root_cause else "Under investigation",
            impact="Operational impact being assessed",
            urgency=urgency if urgency in ["Low", "Medium", "High", "Critical"] else "Medium",
            affected_systems=affected_systems[:10]  # Limit to 10 systems
        )
    
    def _extract_key_phrases(self, description: str) -> List[str]:
        """Extract key identifying phrases from description for better matching"""
        import re
        phrases = []
        
        # Extract specific identifiers (Partner-X, container IDs, error codes)
        partner_match = re.search(r'Partner-[A-Z]', description, re.IGNORECASE)
        if partner_match:
            phrases.append(partner_match.group())
        
        # Extract qualifier patterns
        qualifier_match = re.search(r"qualifier\s+'([^']+)'", description, re.IGNORECASE)
        if qualifier_match:
            phrases.append(f"qualifier '{qualifier_match.group(1)}'")
        
        # Extract segment patterns
        segment_match = re.search(r'in\s+(\w{3})\s+segment', description, re.IGNORECASE)
        if segment_match:
            phrases.append(f"{segment_match.group(1)} segment")
        
        # Extract message types
        edifact_match = re.search(r'EDIFACT\s+(\w+)', description, re.IGNORECASE)
        if edifact_match:
            phrases.append(f"EDIFACT {edifact_match.group(1)}")
        
        return phrases
    
    async def generate_resolution_plan_async(self, description: str, analysis: IncidentAnalysis = None, 
                                            knowledge_entries: List[KnowledgeBase] = None, 
                                            training_examples: List[TrainingData] = None, db=None) -> dict:
        """
        Implements the query flow:
        - Extract error type from description
        - Search both knowledge base and incident cases
        - Return ALL matching solutions with usefulness counts
        - Sort by usefulness count (highest first)
        - No AI enhancement, just direct database results
        """
        from app.services.incident_analyzer import extract_error_type
        error_type = extract_error_type(description)
        logger.info(f"[Query Flow] Extracted error type: {error_type}")

        # Search knowledge base for matches
        kb_matches = []
        if db:
            from app.services.knowledge_base_service import KnowledgeBaseService
            kb_service = KnowledgeBaseService(db)
            kb_matches = kb_service.search_knowledge(error_type)
            # Also search with full description for broader matches
            kb_matches += [e for e in kb_service.search_knowledge(description) if e not in kb_matches]
            
            # Enhanced search: try broader category terms if specific error_type yields no results
            if len(kb_matches) == 0:
                broader_terms = []
                if "edi" in error_type.lower():
                    broader_terms = ["EDI", "EDI/API", "message", "parsing"]
                elif "container" in error_type.lower():
                    broader_terms = ["Container", "CNTR", "duplicate"]
                elif "vessel" in error_type.lower():
                    broader_terms = ["Vessel", "ship", "arrival"]
                
                for term in broader_terms:
                    matches = kb_service.search_knowledge(term)
                    kb_matches += [e for e in matches if e not in kb_matches]
                    if len(kb_matches) >= 5:  # Limit to avoid too many results
                        break
        elif knowledge_entries:
            kb_matches = knowledge_entries

        # Search training data for matches
        td_matches = []
        if db:
            from app.services.training_data_service import TrainingDataService
            td_service = TrainingDataService(db)
            # First try exact phrase matching from description
            key_phrases = self._extract_key_phrases(description)
            for phrase in key_phrases:
                matches = td_service.search_training_data(phrase)
                td_matches += [e for e in matches if e not in td_matches]
            # Then add error_type matches
            td_matches += [e for e in td_service.search_training_data(error_type) if e not in td_matches]
            
            # Enhanced search: try broader category terms if specific error_type yields no results
            if len(td_matches) == 0:
                broader_terms = []
                if "edi" in error_type.lower():
                    broader_terms = ["EDI", "EDI/API"]
                elif "container" in error_type.lower():
                    broader_terms = ["Container", "Container Booking", "Container Report"]
                elif "vessel" in error_type.lower():
                    broader_terms = ["Vessel"]
                
                for term in broader_terms:
                    matches = td_service.search_training_data(term)
                    td_matches += [e for e in matches if e not in td_matches]
                    if len(td_matches) >= 5:  # Limit to avoid too many results
                        break
        elif training_examples:
            td_matches = training_examples

        # Combine all matches with relevance scoring
        all_matches = []
        
        # Calculate relevance scores for KB entries
        for entry in kb_matches:
            # KB entries are templates - give lower base score
            relevance_score = 10 + entry.usefulness_count
            all_matches.append({
                "order": len(all_matches) + 1,
                "title": entry.title,
                "description": entry.content,
                "type": entry.type if entry.type else "Resolution",
                "source": "Knowledge Base",
                "knowledge_base_id": entry.id,
                "training_data_id": None,
                "usefulness_count": entry.usefulness_count,
                "priority": entry.priority,
                "category": entry.category,
                "relevance_score": relevance_score
            })
        
        # Calculate relevance scores for training data (actual cases)
        description_lower = description.lower()
        for example in td_matches:
            # Training data are actual cases - give higher base score
            relevance_score = 100 + int(getattr(example, 'usefulness_count', 0) or 0)
            
            # Boost score for exact phrase matches
            incident_lower = (example.incident_description or "").lower()
            for phrase in key_phrases:
                if phrase.lower() in incident_lower:
                    relevance_score += 50  # Significant boost for exact phrase match
            
            # Boost for matching specific identifiers (Partner-X, specific qualifiers)
            import re
            if re.search(r'Partner-[A-Z]', description):
                desc_partner = re.search(r'Partner-([A-Z])', description)
                incident_partner = re.search(r'Partner-([A-Z])', example.incident_description or "")
                if desc_partner and incident_partner and desc_partner.group(1) == incident_partner.group(1):
                    relevance_score += 100  # Major boost for exact partner match
            
            # Training data display - show Solution (expected_root_cause) and SOP (notes)
            solution = example.expected_root_cause or ""
            sop = example.notes or ""
            
            # Format: Solution first, then SOP if available
            if solution and sop:
                formatted_description = f"Solution: {solution}\n\nSOP: {sop}"
            elif solution:
                formatted_description = f"Solution: {solution}"
            elif sop:
                formatted_description = f"SOP: {sop}"
            else:
                formatted_description = example.incident_description
            
            all_matches.append({
                "order": len(all_matches) + 1,
                "title": f"Case - {example.category}",
                "description": formatted_description,
                "type": "Resolution",
                "source": "Training Data",
                "knowledge_base_id": None,
                "training_data_id": example.id,
                "usefulness_count": int(getattr(example, 'usefulness_count', 0) or 0),
                "category": example.category,
                "relevance_score": relevance_score
            })

        # Sort all matches:
        # 1. First by usefulness_count (highest first) - prioritize proven solutions
        # 2. Then by relevance_score (highest first) - for tiebreaking and sorting within same usefulness level
        sorted_matches = sorted(all_matches, key=lambda x: (x.get("usefulness_count", 0), x.get("relevance_score", 0)), reverse=True)

        # Return all matches for user selection
        return {
            "summary": f"Found {len(sorted_matches)} matching solutions for '{error_type}' (sorted by usefulness, then relevance)",
            "steps": sorted_matches
        }
    
    def _build_steps_from_database(self, knowledge_entries: List[KnowledgeBase] = None, 
                                   training_examples: List[TrainingData] = None) -> List[dict]:
        """Build resolution steps from database solutions (already sorted by usefulness)"""
        steps = []
        
        # Add knowledge base solutions (highest usefulness first)
        if knowledge_entries:
            for idx, entry in enumerate(knowledge_entries[:5], 1):  # Top 5 knowledge entries
                # Show FULL content directly from database (no truncation or AI enhancement)
                description = entry.content
                
                steps.append({
                    "order": len(steps) + 1,
                    "title": entry.title,
                    "description": description,
                    "type": entry.type if entry.type else "Resolution",
                    "source": "Knowledge Base",
                    "knowledge_base_id": entry.id,
                    "training_data_id": None,
                    "usefulness_count": entry.usefulness_count,
                    "priority": entry.priority,
                    "category": entry.category
                })
        
        # Add training data solutions (highest usefulness first)
        if training_examples:
            for idx, example in enumerate(training_examples[:3], 1):  # Top 3 training examples
                # Show FULL content directly from database (no truncation)
                desc_text = example.expected_root_cause or example.incident_description
                description = desc_text
                
                steps.append({
                    "order": len(steps) + 1,
                    "title": f"{example.expected_incident_type} - {example.category}",
                    "description": description,
                    "type": "Resolution",
                    "source": "Training Data",
                    "knowledge_base_id": None,
                    "training_data_id": example.id,
                    "usefulness_count": int(getattr(example, 'usefulness_count', 0) or 0),
                    "category": example.category
                })
        
        return steps

    def _format_database_solutions(self, database_steps: List[dict], incident_type: str) -> dict:
        """Format database solutions into resolution plan without AI enhancement"""
        if not database_steps:
            return self._create_fallback_resolution_plan(incident_type)
        
        # Sort by usefulness count (already sorted, but ensure)
        sorted_steps = sorted(database_steps, key=lambda x: x.get("usefulness_count", 0), reverse=True)
        
        formatted_steps = []
        for idx, step in enumerate(sorted_steps, 1):
            formatted_steps.append({
                "order": idx,
                "title": step.get("title", ""),
                "description": step["description"],
                "type": step["type"],
                "query": "",
                "source": step.get("source", "Database"),
                "knowledge_base_id": step.get("knowledge_base_id"),
                "training_data_id": step.get("training_data_id"),
                "usefulness_count": step.get("usefulness_count", 0),
                "category": step.get("category", "")
            })
        
        return {
            "summary": f"Resolution plan for {incident_type} based on {len(formatted_steps)} proven solutions from database (sorted by usefulness)",
            "steps": formatted_steps
        }

    def _create_enhanced_resolution_prompt(self, description: str, analysis: IncidentAnalysis, 
                                          database_steps: List[dict]) -> str:
        """Create prompt for AI to enhance database solutions"""
        
        db_solutions = "\n".join([
            f"{idx}. [{step.get('source', 'DB')}] (Usefulness: {step.get('usefulness_count', 0)}) {step['description']}"
            for idx, step in enumerate(database_steps, 1)
        ])
        
        return f"""Based on this maritime operations incident, organize and enhance the following PROVEN SOLUTIONS from our database:

INCIDENT: {description}

ANALYSIS:
- Type: {analysis.incident_type}
- Root Cause: {analysis.root_cause}
- Urgency: {analysis.urgency}

PROVEN DATABASE SOLUTIONS (sorted by usefulness count):
{db_solutions if db_solutions else "No database solutions available - create generic steps"}

Your task: Organize these proven solutions into a clear, actionable step-by-step plan. You can:
- Reorder steps for logical flow
- Break complex solutions into smaller steps
- Add transitional steps if needed
- Clarify technical terms
- DO NOT ignore high-usefulness solutions

Return JSON format:
{{
    "summary": "Brief summary emphasizing database-proven solutions",
    "steps": [
        {{
            "order": 1,
            "description": "Clear action based on database solution",
            "type": "Analysis|Investigation|Resolution|Verification",
            "query": "Any diagnostic query if applicable"
        }}
    ]
}}
"""

    def _create_resolution_prompt(self, description: str, analysis: IncidentAnalysis) -> str:
        """Legacy method - kept for backward compatibility"""
        return self._create_enhanced_resolution_prompt(description, analysis, [])
    
    def _parse_resolution_response(self, ai_response: str, incident_type: str) -> dict:
        """Parse AI response into resolution plan"""
        try:
            import json
            # Extract JSON from response
            start = ai_response.find('{')
            end = ai_response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = ai_response[start:end]
                data = json.loads(json_str)
                
                return {
                    "summary": data.get("summary", f"AI-generated resolution plan for {incident_type}"),
                    "steps": data.get("steps", [])
                }
        except Exception as ex:
            logger.error(f"Error parsing resolution response: {ex}")
        
        return self._create_fallback_resolution_plan(incident_type)
    
    def _create_fallback_resolution_plan(self, incident_type: str) -> dict:
        """Create fallback resolution plan when AI is not available"""
        return {
            "summary": f"Structured resolution approach for {incident_type} incident",
            "steps": [
                {
                    "order": 1,
                    "description": "Gather additional incident details and verify system status",
                    "type": "Analysis",
                    "query": "SELECT status FROM system_health WHERE component = 'portnet'"
                },
                {
                    "order": 2,
                    "description": "Identify specific failure points and affected processes", 
                    "type": "Investigation",
                    "query": "SELECT * FROM error_logs WHERE timestamp >= NOW() - INTERVAL 1 HOUR"
                },
                {
                    "order": 3,
                    "description": "Implement targeted fix based on investigation findings",
                    "type": "Resolution",
                    "query": "Apply appropriate system restart or configuration update"
                },
                {
                    "order": 4,
                    "description": "Verify resolution and monitor system stability",
                    "type": "Verification", 
                    "query": "SELECT COUNT(*) FROM error_logs WHERE timestamp >= NOW() - INTERVAL 5 MINUTE"
                }
            ]
        }