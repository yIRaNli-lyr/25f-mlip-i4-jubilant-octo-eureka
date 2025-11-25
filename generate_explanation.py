"""
Diabetic Retinopathy Explanation Generator
Generates HTML reports and PDF explanations for patient retina scans
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import gc
from datetime import datetime
from pathlib import Path

# Optional: weasyprint for PDF generation
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    print("Note: weasyprint not installed. PDF generation will be skipped.")
    print("To install: pip install weasyprint")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DRExplainer:
    """Diabetic Retinopathy Explainer - generates predictions and explanations"""
    
    def __init__(self, model_path, template_path='explanation_template.html'):
        """
        Initialize the explainer
        
        Args:
            model_path: Path to the trained model (.h5 file)
            template_path: Path to the HTML template
        """
        print(f"Loading model from {model_path}...")
        # Load model without compiling to avoid optimizer compatibility issues
        self.model = keras.models.load_model(model_path, compile=False)
        self.template_path = template_path
        
        # Class names for diabetic retinopathy
        self.class_names = [
            'No DR',
            'Mild DR',
            'Moderate DR',
            'Severe DR',
            'Proliferative DR'
        ]
        
        # Target image size (adjust based on your model)
        self.img_size = (320, 320)
        
        print("Model loaded successfully!")
    
    def preprocess_image(self, img_path):
        """Load and preprocess an image for prediction"""
        img = keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def generate_activation_heatmap(self, img_array, last_conv_layer_name='activation_49'):
        """
        Generate activation-based heatmap (CPU-friendly, no gradients)
        Uses the activation layer after the last conv block
        
        Args:
            img_array: Preprocessed image array
            last_conv_layer_name: Name of the activation layer after last conv
        
        Returns:
            heatmap: numpy array of the heatmap
        """
        # Force CPU execution to avoid GPU memory issues
        with tf.device('/CPU:0'):
            # Clear any previous session
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Create a model that outputs the activation of the last conv layer
            last_conv_layer = self.model.get_layer(last_conv_layer_name)
            activation_model = keras.Model(
                inputs=self.model.input,
                outputs=last_conv_layer.output
            )
            
            # Get the activations
            activations = activation_model.predict(img_array, verbose=0)
            
            # Average across all filters (channels)
            heatmap = np.mean(activations[0], axis=-1)
            
            # Normalize to [0, 1]
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) != 0:
                heatmap /= np.max(heatmap)
            
            # Resize heatmap to match image size
            zoom_factor = np.array(self.img_size) / np.array(heatmap.shape)
            heatmap_resized = zoom(heatmap, zoom_factor, order=1)
            
            return heatmap_resized
    
    def save_heatmap_overlay(self, img_path, heatmap, output_path, draw_boxes=True):
        """
        Save image with bounding boxes highlighting important regions
        
        Args:
            img_path: Path to original image
            heatmap: Generated heatmap array
            output_path: Where to save the annotated image
            draw_boxes: Whether to draw bounding boxes (False for healthy cases)
        """
        import cv2
        
        # Load original image
        img = keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
        img_array = keras.preprocessing.image.img_to_array(img).astype(np.uint8)
        
        # Debug: check heatmap values
        print(f"  Heatmap min: {heatmap.min():.4f}, max: {heatmap.max():.4f}, mean: {heatmap.mean():.4f}")
        
        if not draw_boxes:
            print(f"  No disease detected - skipping bounding boxes")
            # Save original image without boxes
            plt.figure(figsize=(8, 8))
            plt.imshow(img_array)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            print(f"Saved image (no boxes) to {output_path}")
            return
        
        # Find regions with high activation (top 30% - more precise)
        threshold = np.percentile(heatmap, 70)  # Top 30%
        high_activation = heatmap > threshold
        
        # Use smaller kernel for tighter contours
        kernel = np.ones((3,3), np.uint8)  # Smaller kernel = tighter boxes
        high_activation_uint8 = (high_activation * 255).astype(np.uint8)
        high_activation_uint8 = cv2.morphologyEx(high_activation_uint8, cv2.MORPH_CLOSE, kernel)
        high_activation_uint8 = cv2.morphologyEx(high_activation_uint8, cv2.MORPH_OPEN, kernel)
        
        # Find contours of high activation regions
        contours, _ = cv2.findContours(high_activation_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Found {len(contours)} contours total")
        
        # Draw bounding boxes around important regions
        box_count = 0
        # Sort contours by area (largest first) and take top 5
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in contours_sorted:
            area = cv2.contourArea(contour)
            print(f"  Contour area: {area:.1f} pixels ({area/(self.img_size[0]*self.img_size[1])*100:.2f}% of image)")
            
            # Only draw significant regions (1% or more)
            if area > (self.img_size[0] * self.img_size[1] * 0.01):
                x, y, w, h = cv2.boundingRect(contour)
                # Draw thin red rectangle (2 pixels)
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Add smaller label
                label = 'Problem'
                cv2.putText(img_array, label, (x+3, y+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4)  # White outline
                cv2.putText(img_array, label, (x+3, y+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Red text
                box_count += 1
        
        print(f"  Drew {box_count} bounding boxes")
        
        # Save annotated image
        plt.figure(figsize=(8, 8))
        plt.imshow(img_array)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        print(f"Saved annotated image to {output_path}")
    
    def predict(self, img_path):
        """
        Make prediction on an image
        
        Args:
            img_path: Path to the image file
        
        Returns:
            prediction: Class index
            confidence: Confidence score
            class_name: Predicted class name
        """
        img_array = self.preprocess_image(img_path)
        predictions = self.model.predict(img_array, verbose=0)
        
        prediction = np.argmax(predictions[0])
        confidence = predictions[0][prediction]
        class_name = self.class_names[prediction]
        
        return prediction, confidence, class_name
    
    def generate_explanation_text(self, class_name, confidence):
        """Generate simple, conversational explanation"""
        explanations = {
            'No DR': "Your eyes look good. Keep monitoring your blood sugar.",
            'Mild DR': "We see some tiny red spots. Early stage - still easy to manage.",
            'Moderate DR': "Some bleeding and vessel damage detected. You should see a doctor.",
            'Severe DR': "Significant damage found. Please schedule an eye exam soon.",
            'Proliferative DR': "Advanced stage detected. Urgent - see a specialist immediately."
        }
        
        explanation = explanations.get(class_name, "Unable to assess")
        
        return explanation
    
    def generate_html_report(self, patient_info, img_path, output_html_path, healthy_retina_path='0a38b552372d.png'):
        """
        Generate HTML report for a patient
        
        Args:
            patient_info: Dictionary with keys: 'id', 'gender', 'age', etc.
            img_path: Path to patient's retina image
            output_html_path: Where to save the HTML report
            healthy_retina_path: Path to the healthy retina reference image
        
        Returns:
            output_html_path: Path to generated HTML
        """
        print(f"\nGenerating report for patient {patient_info['id']}...")
        
        # Make prediction
        prediction, confidence, class_name = self.predict(img_path)
        print(f"  Prediction: {class_name} ({confidence*100:.1f}% confidence)")
        
        # Generate heatmap and annotated image
        img_array = self.preprocess_image(img_path)
        heatmap = self.generate_activation_heatmap(img_array)
        
        # Save heatmap overlay (only draw boxes if disease detected)
        heatmap_path = output_html_path.replace('.html', '_heatmap.png')
        draw_boxes = (class_name != 'No DR')  # Only draw boxes if not healthy
        self.save_heatmap_overlay(img_path, heatmap, heatmap_path, draw_boxes=draw_boxes)
        
        # Copy healthy retina image to output directory if it exists
        output_dir = os.path.dirname(output_html_path)
        if os.path.exists(healthy_retina_path):
            import shutil
            healthy_copy_path = os.path.join(output_dir, os.path.basename(healthy_retina_path))
            shutil.copy2(healthy_retina_path, healthy_copy_path)
            print(f"  Copied healthy retina reference to {healthy_copy_path}")
        
        # Generate explanation text
        explanation_text = self.generate_explanation_text(class_name, confidence)
        
        # Read template
        with open(self.template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Replace placeholders
        html_content = html_content.replace('PXX', patient_info['id'])
        html_content = html_content.replace('GXX', patient_info['gender'])
        html_content = html_content.replace('DXX', class_name)
        html_content = html_content.replace('[DATE]', datetime.now().strftime('%Y-%m-%d'))
        
        # Replace header based on diagnosis
        if class_name == 'No DR':
            # Healthy patient - green background, white text
            html_content = html_content.replace(
                '.header {\n  background-color: #D32F2F;',
                '.header {\n  background-color: #4caf50;'
            )
            html_content = html_content.replace(
                '<h1>SCREENING RESULT: REFERRAL RECOMMENDED</h1>',
                '<h1>SCREENING RESULT: NO ABNORMALITIES DETECTED</h1>'
            )
            html_content = html_content.replace(
                '<h2>Potential signs of Diabetic Retinopathy detected</h2>',
                '<h2>Your retinal screening shows healthy results</h2>'
            )
            html_content = html_content.replace(
                '<div class="cta">Please schedule a dilated eye exam within 2 weeks</div>',
                '<div class="cta" style="background-color: #ffffff; color: #4caf50;">Continue routine annual eye exams</div>'
            )
            
            # Replace Next Steps for healthy patients
            healthy_next_steps = '''  <h2>Next Steps</h2>
  <ol>
    <li><strong>Keep Up Good Control:</strong> Continue managing your blood sugar levels as recommended by your doctor.</li>
    <li><strong>Annual Screening:</strong> Schedule your next eye screening in 12 months to monitor your eye health.</li>
    <li><strong>Stay Healthy:</strong> Maintain a healthy diet, exercise regularly, and follow your diabetes care plan.</li>
  </ol>'''
            html_content = html_content.replace(
                '''  <h2>Next Steps</h2>
  <ol>
    <li><strong>Call an Eye Doctor:</strong> Search for an "Ophthalmologist" or "Retina Specialist" near you.</li>
    <li><strong>Bring This Sheet:</strong> Show the doctor the "Physician's Note" below.</li>
    <li><strong>Don't Wait:</strong> Make the appointment today. Early action saves your sight.</li>
  </ol>''',
                healthy_next_steps
            )
            
            # Replace Physician's Note for healthy patients
            healthy_physician_note = '''  <h2>Physician's Note</h2>
  <p>
    <strong>To the attending Physician:</strong><br><br>
    This patient underwent a non-mydriatic fundus screening via an AI-assisted smartphone device. 
    The system detected <strong>NO signs of Diabetic Retinopathy</strong>. 
    <br><br>
    Patient is cleared for routine annual screening. Continue regular diabetes management and monitoring.
  </p>'''
            html_content = html_content.replace(
                '''  <h2>Physician's Note</h2>
  <p>
    <strong>To the attending Ophthalmologist:</strong><br><br>
    This patient underwent a non-mydriatic fundus screening via an AI-assisted smartphone device. 
    The system detected high-probability indicators of Referable Diabetic Retinopathy (Sensitivity: 97%). 
    <br><br>
    Please perform a comprehensive dilated eye exam to confirm diagnosis and determine treatment plan.
  </p>''',
                healthy_physician_note
            )
            
            # Remove "Why Act Now?" section for healthy patients
            import re
            why_act_pattern = r'<!-- Why Act Now Section -->.*?</div>\s*</div>\s*</div>'
            html_content = re.sub(why_act_pattern, '', html_content, flags=re.DOTALL)
            
            # Remove "The Cost of Waiting vs. Acting" section for healthy patients
            financial_pattern = r'<!-- Financial Section -->.*?</div>\s*</div>\s*</div>'
            html_content = re.sub(financial_pattern, '', html_content, flags=re.DOTALL)
            
            # Replace image description for healthy patients
            html_content = html_content.replace(
                '<p><strong>"Your Screening Scan"</strong> - The highlighted areas show potential micro-aneurysms (bleeding spots). These are early warning signs of damage.</p>',
                '<p><strong>"Your Screening Scan"</strong> - No abnormal areas detected. Retinal vessels appear clear and healthy.</p>'
            )
        else:
            # Patient with disease - red background, white text (already default in template)
            pass
        
        # Replace image paths (use basenames for relative paths)
        heatmap_filename = os.path.basename(heatmap_path)
        html_content = html_content.replace('IMAGE_FOR_DISPLAY.png', heatmap_filename)
        
        # Replace explanation
        html_content = html_content.replace('EXPLANATION', explanation_text)
        
        # Save HTML
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"  HTML report saved to {output_html_path}")
        
        return output_html_path


def generate_pdf_from_htmls(html_paths, output_pdf_path):
    """
    Combine multiple HTML reports into a single PDF
    
    Args:
        html_paths: List of HTML file paths
        output_pdf_path: Where to save the combined PDF
    """
    if not WEASYPRINT_AVAILABLE:
        print("\nPDF generation skipped: weasyprint not installed")
        print("   You can still view the HTML reports in a browser")
        print("   To install weasyprint: pip install weasyprint")
        return
    
    print(f"\nGenerating PDF with {len(html_paths)} reports...")
    
    # Read all HTML contents
    combined_html = "<html><head><style>body { page-break-after: always; }</style></head><body>"
    
    for i, html_path in enumerate(html_paths):
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract body content (between <body> and </body>)
            body_start = content.find('<body>') + 6
            body_end = content.find('</body>')
            body_content = content[body_start:body_end]
            
            # Extract style content
            style_start = content.find('<style>') + 7
            style_end = content.find('</style>')
            style_content = content[style_start:style_end]
            
            if i == 0:
                # Add styles once at the beginning
                combined_html = f"<html><head><style>{style_content}</style></head><body>"
            
            combined_html += body_content
            
            # Add page break except for last page
            if i < len(html_paths) - 1:
                combined_html += '<div style="page-break-after: always;"></div>'
    
    combined_html += "</body></html>"
    
    # Convert to PDF using weasyprint
    try:
        weasyprint.HTML(string=combined_html).write_pdf(output_pdf_path)
        print(f"PDF saved to {output_pdf_path}")
        print(f"Successfully created {output_pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")


def main():
    """
    Main function with command-line argument support
    
    Usage examples:
    1. Single patient (command line):
       python generate_explanation.py --id P001 --gender Male --age 58 --image patient.png
    
    2. Multiple patients (edit patients list in code):
       python generate_explanation.py
    
    3. Help:
       python generate_explanation.py --help
    """
    parser = argparse.ArgumentParser(
        description='Generate diabetic retinopathy explanation reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single patient
  python generate_explanation.py --id P001 --gender Male --age 58 --image patient.png
  
  # Multiple patients from list (edit code)
  python generate_explanation.py
        """
    )
    
    # Command line arguments for single patient
    parser.add_argument('--id', type=str, help='Patient ID')
    parser.add_argument('--gender', type=str, help='Patient gender')
    parser.add_argument('--age', type=int, help='Patient age')
    parser.add_argument('--image', type=str, help='Path to patient retina image')
    
    args = parser.parse_args()
    
    # ===== CONFIGURATION =====
    MODEL_PATH = 'model.h5'
    TEMPLATE_PATH = 'explanation_template.html'
    OUTPUT_DIR = 'patient_reports'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ===== PATIENT DATA =====
    # Check if command line arguments provided
    if args.id and args.gender and args.age and args.image:
        # Single patient from command line
        patients = [{
            'id': args.id,
            'gender': args.gender,
            'age': args.age,
            'image_path': args.image
        }]
        print(f"Processing single patient from command line: {args.id}")
    else:
        # Multiple patients from list (edit here as needed)
        patients = [
            {
                'id': 'P001',
                'gender': 'Male',
                'age': 58,
                'image_path': '1b32e1d775ea.png'
            },
            # Add more patients:
            # {
            #     'id': 'P002',
            #     'gender': 'Female',
            #     'age': 62,
            #     'image_path': 'patient2.png'
            # },
        ]
        print(f"Processing {len(patients)} patient(s) from list")
    
    # ===== INITIALIZE EXPLAINER =====
    explainer = DRExplainer(MODEL_PATH, TEMPLATE_PATH)
    
    # ===== GENERATE REPORTS FOR EACH PATIENT =====
    html_paths = []
    
    for patient in patients:
        output_html = os.path.join(OUTPUT_DIR, f"report_{patient['id']}.html")
        
        try:
            explainer.generate_html_report(patient, patient['image_path'], output_html)
            html_paths.append(output_html)
        except Exception as e:
            print(f"Error generating report for {patient['id']}: {e}")
    
    # ===== COMBINE INTO PDF (optional) =====
    if html_paths and len(html_paths) > 1:
        output_pdf = 'explanation_local.pdf'
        generate_pdf_from_htmls(html_paths, output_pdf)
    
    print("\n" + "="*50)
    print("All reports generated successfully!")
    print(f"HTML reports: {OUTPUT_DIR}/")
    if len(html_paths) > 1:
        print(f"Combined PDF: explanation_local.pdf")
    print("="*50)


if __name__ == '__main__':
    main()
