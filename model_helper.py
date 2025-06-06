import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os
import logging
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

trained_model = None
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', "sheep", 'spider', 'squirrel']


# Load the pre-trained ResNet model
class AnimalClassifierResNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True            
            
        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def load_model(model_path="saved_model.pth"):
    """Load the trained model with error handling"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = AnimalClassifierResNet(num_classes=len(class_names))
        
        # Load model with proper device mapping
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, map_location=device)
        
        model.load_state_dict(state_dict)
        model.eval()
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def validate_image(image_path):
    """Validate image file before processing"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if os.path.getsize(image_path) == 0:
        raise ValueError("Image file is empty")
    
    # Check if file is a valid image
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")


def preprocess_image(image_path):
    """Preprocess image with error handling"""
    try:
        validate_image(image_path)
        
        # Try to open the image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Direct image opening failed: {e}. Trying alternative method...")
            # Alternative method: read as bytes first
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise


def predict(image_path, model_path="saved_model.pth"):
    """Make prediction with comprehensive error handling"""
    global trained_model
    
    try:
        # Load model if not already loaded
        if trained_model is None:
            trained_model = load_model(model_path)
        
        # Preprocess image
        image_tensor = preprocess_image(image_path)
        
        # Make prediction
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trained_model.to(device)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            output = trained_model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_animal = class_names[predicted_class.item()]
            confidence_score = confidence.item()
            
            logger.info(f"Prediction: {predicted_animal} (confidence: {confidence_score:.4f})")
            
            return {
                'prediction': predicted_animal,
                'confidence': confidence_score
            }
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


def predict_simple(image_path, model_path="saved_model.pth"):
    """Simple prediction function that returns just the class name"""
    try:
        result = predict(image_path, model_path)
        return result['prediction']
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return f"Error: {str(e)}"


def get_model_info():
    """Get information about the loaded model"""
    global trained_model
    if trained_model is not None:
        total_params = sum(p.numel() for p in trained_model.parameters())
        trainable_params = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'classes': class_names,
            'num_classes': len(class_names)
        }
    else:
        return "Model not loaded"


# Example usage and testing
if __name__ == "__main__":
    # Test the model loading
    try:
        test_result = predict_simple("test_image.jpg")
        print(f"Test prediction: {test_result}")
    except Exception as e:
        print(f"Test failed: {e}")