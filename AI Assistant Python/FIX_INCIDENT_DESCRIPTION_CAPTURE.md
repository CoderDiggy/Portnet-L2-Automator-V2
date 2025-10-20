# Fix: Incident Description Now Captured in Solution Feedback

## Problem
When users clicked the thumbs-up button, the feedback was being stored with "Unknown incident" instead of the actual incident description that the user entered.

## Root Cause
1. The incident description was being passed to the HTML template correctly
2. However, special characters (quotes, apostrophes, etc.) in the incident description were not properly escaped for HTML attributes
3. This caused the `data-incident-description` attribute to be malformed or truncated
4. JavaScript was checking `if (incidentDescription)` before sending it, so if it was null/empty, it wouldn't send it
5. The API endpoint had a fallback to "Unknown incident"

## Solution

### 1. Updated Template (results.html)
- Added Jinja2's `| e` filter to properly escape HTML special characters
- This ensures quotes, apostrophes, and other special characters don't break the HTML attribute

```html
data-incident-description="{{ result.incident.description | e }}"
data-step-description="{{ step.description | e }}"
```

### 2. Updated JavaScript (results.js)
- Changed to ALWAYS send incident_description (with fallback)
- Before: `if (incidentDescription) formData.append('incident_description', incidentDescription);`
- After: `formData.append('incident_description', incidentDescription);` (always sends it)
- Added default fallback: `const incidentDescription = this.getAttribute('data-incident-description') || 'Unknown incident';`

### 3. Updated API Endpoint (simple_main.py)
- Added logging to track when empty incident descriptions are received
- Added proper handling of whitespace and empty strings
- Now updates existing feedback entries with better incident descriptions if available
- Strips whitespace before storing

```python
# Log for debugging
if not incident_description or incident_description.strip() == "":
    logger.warning(f"Received empty incident_description...")
else:
    logger.info(f"Received incident_description: {incident_description[:100]}...")

# Clean and use the description
final_incident_description = incident_description.strip() if incident_description and incident_description.strip() else "Unknown incident"
```

## Testing
To verify the fix works:

1. Start the application
2. Go to /analyze
3. Enter an incident description (try one with quotes like: `Container "ABC123" is stuck`)
4. Click Analyze
5. Click the thumbs-up button on any solution
6. Run: `python query_solution_feedback.py`
7. Verify the incident_description now shows your actual input, not "Unknown incident"

## Files Changed
- `app/templates/results.html` - Added HTML escaping
- `static/js/results.js` - Always send incident description
- `simple_main.py` - Improved logging and handling

## Benefits
- Feedback now correctly tracks which solutions work for which specific problems
- Better data for analyzing solution effectiveness
- Can see exactly what incident descriptions lead to which solutions being marked useful
- HTML special characters no longer break the functionality
