from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import openai
import PyPDF2
import io
import json
import re
from werkzeug.utils import secure_filename
import os
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, origins=["*"])

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_stream):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting PDF text: {str(e)}")
        return None

def extract_text_from_txt(file_stream):
    """Extract text from TXT file"""
    try:
        content = file_stream.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        return content.strip()
    except Exception as e:
        logging.error(f"Error extracting TXT text: {str(e)}")
        return None

def clean_text(text):
    """Clean and preprocess text"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def generate_flashcards_with_openai(content: str, api_key: str, num_cards: int = 15) -> List[Dict]:
    """Generate flashcards using OpenAI GPT"""
    try:
        # Set up OpenAI client
        openai.api_key = api_key
        
        prompt = f"""
        Create {num_cards} high-quality flashcards from the following educational content. 
        Each flashcard should have a clear, concise question and a comprehensive answer.
        
        Guidelines:
        - Questions should test understanding, not just memorization
        - Include different types of questions: definitions, explanations, applications, comparisons
        - Answers should be detailed but concise (2-4 sentences)
        - Cover the most important concepts from the content
        - Make questions specific and unambiguous
        
        Content:
        {content[:4000]}  # Limit content to avoid token limits
        
        Return the flashcards in this exact JSON format:
        [
            {{"question": "Question text here", "answer": "Answer text here"}},
            {{"question": "Question text here", "answer": "Answer text here"}}
        ]
        
        Only return the JSON array, no additional text.
        """
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert educator who creates high-quality flashcards for learning. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract JSON from response
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            flashcards_data = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from text
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                flashcards_data = json.loads(json_match.group())
            else:
                raise ValueError("Could not extract valid JSON from OpenAI response")
        
        # Add IDs to flashcards
        flashcards = []
        for i, card in enumerate(flashcards_data[:num_cards]):  # Ensure we don't exceed requested number
            flashcards.append({
                "id": i + 1,
                "question": card.get("question", "").strip(),
                "answer": card.get("answer", "").strip()
            })
        
        return flashcards
        
    except Exception as e:
        logging.error(f"Error generating flashcards with OpenAI: {str(e)}")
        raise

@app.route('/api/generate-flashcards', methods=['POST'])
def generate_flashcards():
    """Generate flashcards from text or file input"""
    try:
        # Get API key from request
        api_key = request.form.get('api_key') or request.json.get('api_key') if request.is_json else None
        if not api_key:
            return jsonify({"error": "OpenAI API key is required"}), 400
        
        content = ""
        
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_extension = filename.rsplit('.', 1)[1].lower()
                
                # Extract text based on file type
                if file_extension == 'pdf':
                    content = extract_text_from_pdf(file.stream)
                elif file_extension == 'txt':
                    content = extract_text_from_txt(file.stream)
                
                if not content:
                    return jsonify({"error": "Could not extract text from file"}), 400
        
        # Get text input
        text_input = request.form.get('text', '').strip()
        if text_input:
            content += "\n\n" + text_input if content else text_input
        
        if not content:
            return jsonify({"error": "No content provided"}), 400
        
        # Clean and validate content
        content = clean_text(content)
        if len(content) < 50:
            return jsonify({"error": "Content too short. Please provide more detailed content for better flashcards."}), 400
        
        # Generate flashcards using OpenAI
        flashcards = generate_flashcards_with_openai(content, api_key, num_cards=15)
        
        if len(flashcards) < 10:
            return jsonify({"error": "Could not generate minimum required flashcards. Please provide more content."}), 400
        
        return jsonify({
            "success": True,
            "flashcards": flashcards,
            "total_cards": len(flashcards),
            "content_length": len(content)
        })
        
    except Exception as e:
        logging.error(f"Error in generate_flashcards endpoint: {str(e)}")
        return jsonify({"error": f"Failed to generate flashcards: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "flashcard-generator"})

@app.route('/api/validate-key', methods=['POST'])
def validate_openai_key():
    """Validate OpenAI API key"""
    try:
        data = request.get_json()
        api_key = data.get('api_key')
        
        if not api_key:
            return jsonify({"valid": False, "error": "API key is required"}), 400
        
        # Test the API key with a simple request
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        return jsonify({"valid": True})
        
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 400

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("Starting FlashCard Generator API Server...")
    print("Make sure to install required packages:")
    print("pip install flask flask-cors openai PyPDF2 werkzeug")
    print("\nServer will run on http://localhost:5000")
    print("API endpoints:")
    print("- POST /api/generate-flashcards")
    print("- POST /api/validate-key") 
    print("- GET /api/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)