from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
import base64
import threading
import time
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Get environment variables with defaults
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://imageshrinks.netlify.app')
BACKEND_URL = os.environ.get('RENDER_EXTERNAL_URL', 'https://image-compressor-584i.onrender.com')
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'https://imageshrinks.netlify.app,http://localhost:3000').split(',')
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 10485760))  # 10MB default
ALLOWED_FORMATS = os.environ.get('ALLOWED_FORMATS', 'jpg,jpeg,png,webp').split(',')

# Configure CORS properly - single definition
CORS(app, resources={
    r"/*": {
        "origins": CORS_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Keep server alive mechanism
def keep_alive():
    while True:
        try:
            # Use the environment-based URL
            requests.get(f'{BACKEND_URL}/ping')
            app.logger.info("Keep-alive ping successful")
        except Exception as e:
            app.logger.error(f"Keep-alive request failed: {str(e)}")
        time.sleep(840)  # 14 minutes in seconds

@app.route('/ping')
def ping():
    app.logger.info("Ping received")
    return 'pong'

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Server is running", "version": "1.0.1"}), 200

# Add explicit OPTIONS route handlers for each API endpoint
@app.route('/api/compress', methods=['OPTIONS'])
def options_compress():
    return '', 200

@app.route('/api/convert', methods=['OPTIONS'])
def options_convert():
    return '', 200

def compress_image(img, target_size_kb, max_attempts=20):
    """Compress image to target size using binary search for quality."""
    try:
        min_quality = 1
        max_quality = 100  # Increased from 95 to 100 for better quality at larger sizes
        best_quality = max_quality
        best_diff = float('inf')
        best_buffer = None
        
        # Get initial file size
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=100, optimize=True)
        initial_size = buffer.tell() / 1024  # Convert to KB
        
        # If original is smaller than target, return original
        if initial_size <= target_size_kb:
            buffer.seek(0)
            return buffer.getvalue()

        # Convert to RGB if RGBA (PNG) to ensure JPEG compression works
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Resize image if it's too large (max dimension 5000px)
        max_dimension = 5000
        if img.width > max_dimension or img.height > max_dimension:
            ratio = min(max_dimension / img.width, max_dimension / img.height)
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        for attempt in range(max_attempts):
            quality = (min_quality + max_quality) // 2
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            current_size = buffer.tell() / 1024  # Convert to KB

            size_diff = current_size - target_size_kb
            
            # Update best result if this is closer to target size
            if abs(size_diff) < abs(best_diff):
                best_diff = size_diff
                best_quality = quality
                buffer.seek(0)
                best_buffer = buffer.getvalue()

            # Binary search adjustment
            if current_size > target_size_kb:
                max_quality = quality - 1
            else:
                min_quality = quality + 1

            # Break if we're within 2% of target size or if search range is exhausted
            if abs(size_diff) < (target_size_kb * 0.02) or min_quality > max_quality:
                break

        # If we couldn't get close enough to target size, try one final time with exact quality
        if abs(best_diff) > (target_size_kb * 0.05):
            final_buffer = io.BytesIO()
            final_quality = int(best_quality * (target_size_kb / (best_diff + target_size_kb)))
            final_quality = max(1, min(100, final_quality))
            img.save(final_buffer, format='JPEG', quality=final_quality, optimize=True)
            final_size = final_buffer.tell() / 1024
            
            if abs(final_size - target_size_kb) < abs(best_diff):
                final_buffer.seek(0)
                return final_buffer.getvalue()

        return best_buffer
    except Exception as e:
        app.logger.error(f'Error in compress_image: {str(e)}')
        raise

@app.route('/api/compress', methods=['POST'])
def compress():
    try:
        app.logger.info("Compression request received")
        
        # Debug log for request details
        app.logger.info(f"Request headers: {dict(request.headers)}")
        app.logger.info(f"Request form data keys: {list(request.form.keys())}")
        app.logger.info(f"Request files keys: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            app.logger.error("No file provided in request")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if not file.filename:
            app.logger.error("No filename in uploaded file")
            return jsonify({'error': 'No file selected'}), 400

        # Check file size
        file_data = file.read()
        file_size_mb = len(file_data) / (1024 * 1024)
        app.logger.info(f"Received file size: {file_size_mb:.2f}MB")
        
        if len(file_data) > MAX_FILE_SIZE:
            app.logger.error(f"File too large: {file_size_mb:.2f}MB > {MAX_FILE_SIZE/(1024*1024)}MB")
            return jsonify({'error': f'File size should be less than {MAX_FILE_SIZE/(1024*1024)}MB'}), 400

        # Get target size from request
        target_size = request.form.get('target_size', type=int)
        if not target_size or target_size < 10:
            app.logger.error(f"Invalid target size: {target_size}")
            return jsonify({'error': 'Target size must be at least 10KB'}), 400
        elif target_size > 5000:
            app.logger.error(f"Target size too large: {target_size}")
            return jsonify({'error': 'Target size must be less than 5000KB'}), 400

        # Get output format from request
        output_format = request.form.get('format', 'original').lower()
        app.logger.info(f"Requested output format: {output_format}")
        
        if output_format not in ['original', 'jpg', 'jpeg', 'png', 'webp']:
            app.logger.error(f"Invalid format requested: {output_format}")
            return jsonify({'error': 'Invalid output format'}), 400
        
        try:
            # Read the image into memory
            file_stream = io.BytesIO(file_data)
            
            try:
                # Try to open and verify the image
                img = Image.open(file_stream)
                img.verify()
                
                # Reset stream and reopen image for processing
                file_stream.seek(0)
                img = Image.open(file_stream)
                
                app.logger.info(f"Image opened successfully: {img.format} {img.size} {img.mode}")
                
                # Check image dimensions
                if img.width > 5000 or img.height > 5000:
                    app.logger.error(f"Image dimensions too large: {img.width}x{img.height}")
                    return jsonify({'error': 'Image dimensions should be less than 5000x5000'}), 400
                
                # Convert RGBA to RGB if needed (except for PNG and WebP)
                if img.mode == 'RGBA' and output_format not in ['png', 'webp']:
                    app.logger.info("Converting RGBA to RGB")
                    img = img.convert('RGB')
                
                # Get original file size
                orig_buffer = io.BytesIO()
                img.save(orig_buffer, format='JPEG', quality=95)
                original_size = orig_buffer.tell() / 1024  # KB
                
                app.logger.info(f'Processing image - Original: {original_size:.1f}KB, Target: {target_size}KB')
                
                # First compress with JPEG
                compressed_data = compress_image(img, target_size)
                
                if not compressed_data:
                    app.logger.error("Failed to compress image")
                    return jsonify({'error': 'Failed to compress image'}), 500

                app.logger.info(f"Compression successful, size: {len(compressed_data)/1024:.1f}KB")
                
                # Convert to requested format if needed
                if output_format not in ['original', 'jpg', 'jpeg']:
                    app.logger.info(f"Converting to {output_format} format")
                    temp_img = Image.open(io.BytesIO(compressed_data))
                    output = io.BytesIO()
                    
                    if output_format == 'png':
                        temp_img.save(output, format='PNG', optimize=True)
                    elif output_format == 'webp':
                        # Try different WebP qualities to match target size
                        best_webp_quality = 95
                        best_webp_diff = float('inf')
                        best_webp_data = None
                        
                        for quality in range(95, 0, -5):
                            webp_buffer = io.BytesIO()
                            temp_img.save(webp_buffer, format='WebP', quality=quality, method=6)
                            webp_size = webp_buffer.tell() / 1024
                            
                            size_diff = abs(webp_size - target_size)
                            if size_diff < best_webp_diff:
                                best_webp_diff = size_diff
                                best_webp_quality = quality
                                webp_buffer.seek(0)
                                best_webp_data = webp_buffer.getvalue()
                            
                            if webp_size <= target_size:
                                break
                        
                        compressed_data = best_webp_data
                    
                # Log actual compressed size
                final_size = len(compressed_data) / 1024
                app.logger.info(f'Compression complete - Target: {target_size}KB, Actual: {final_size:.1f}KB')

                # Convert binary data to base64
                compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')

                response_data = {
                    'compressed_data': compressed_base64,
                    'original_size': original_size,
                    'compressed_size': final_size
                }
                
                app.logger.info("Sending successful response")
                return jsonify(response_data)

            except (IOError, OSError) as e:
                app.logger.error(f'Image processing error: {str(e)}')
                return jsonify({'error': 'Invalid image file or format'}), 400

        except Exception as e:
            app.logger.error(f'Error reading file: {str(e)}')
            return jsonify({'error': 'Error reading file'}), 400

    except Exception as e:
        app.logger.error(f'Error during compression: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/convert', methods=['POST'])
def convert():
    try:
        app.logger.info("Convert request received")
        
        if 'image' not in request.files:
            app.logger.error("No image provided for conversion")
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        target_format = request.form.get('targetFormat', 'jpeg').lower()
        
        app.logger.info(f"Converting to format: {target_format}")
        
        # Normalize jpg to jpeg
        if target_format == 'jpg':
            target_format = 'jpeg'
        
        if target_format not in ['jpeg', 'png', 'webp']:
            app.logger.error(f"Unsupported format: {target_format}")
            return jsonify({'error': 'Unsupported format'}), 400
        
        # Open and process image
        image = Image.open(file.stream)
        app.logger.info(f"Image opened for conversion: {image.format} {image.size} {image.mode}")
        
        # Convert to RGB if necessary, except for PNG which can handle RGBA
        if target_format != 'png' and image.mode in ('RGBA', 'P'):
            app.logger.info(f"Converting image mode from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Save in new format
        output = io.BytesIO()
        
        # Set format-specific parameters
        save_params = {
            'format': target_format.upper(),
            'quality': 95 if target_format in ['jpeg', 'jpg', 'webp'] else None,
            'optimize': True
        }
        
        # Remove quality parameter for PNG
        if target_format == 'png':
            save_params.pop('quality')
        
        image.save(output, **save_params)
        output.seek(0)
        
        file_size = output.tell() / 1024
        app.logger.info(f"Conversion complete, size: {file_size:.1f}KB")
        
        # Use jpg extension if that's what was requested
        download_ext = 'jpg' if request.form.get('targetFormat') == 'jpg' else target_format
        
        return send_file(
            output,
            mimetype=f'image/{target_format}',
            as_attachment=True,
            download_name=f'converted.{download_ext}'
        )
    
    except Exception as e:
        app.logger.error(f'Error in conversion: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    if not app.debug:
        keep_alive_thread = threading.Thread(target=keep_alive)
        keep_alive_thread.daemon = True
        keep_alive_thread.start()
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
