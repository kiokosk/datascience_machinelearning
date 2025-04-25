import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet152
import numpy as np

class EncoderCNN(models.Model):
    """
    Keras implementation of the EncoderCNN from the PyTorch model.py
    This encoder uses a pretrained ResNet-152 to extract features from images.
    """
    def __init__(self, embed_size):
        """
        Initialize the model by setting up the layers.
          
        Args:
            embed_size: dimension of the feature vectors
        """
        super(EncoderCNN, self).__init__()
        
        # Load pretrained ResNet but exclude the final FC layer
        base_model = ResNet152(include_top=False, weights='imagenet', pooling='avg')
        
        # Freeze the ResNet layers
        for layer in base_model.layers:
            layer.trainable = False
            
        self.resnet = base_model
        self.linear = layers.Dense(embed_size)
        self.bn = layers.BatchNormalization(momentum=0.01)
        
    def call(self, images, training=False):
        """
        Extract feature vectors from input images.
        
        Args:
            images: input images, (batch_size, height, width, channels)
            training: whether the call is in training mode
            
        Returns:
            features: feature vectors, (batch_size, embed_size)
        """
        features = self.resnet(images)
        features = self.linear(features)
        features = self.bn(features, training=training)
        return features


class DecoderRNN(models.Model):
    """
    Keras implementation of the DecoderRNN from the PyTorch model.py
    This decoder takes image features and generates captions.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_seq_length=20):
        """
        Initialize the model by setting up the layers.
        
        Args:
            embed_size: dimension of word embeddings
            hidden_size: dimension of LSTM hidden states
            vocab_size: size of vocabulary
            num_layers: number of LSTM layers
            max_seq_length: maximum sequence length for generation
        """
        super(DecoderRNN, self).__init__()
        
        self.embed = layers.Embedding(vocab_size, embed_size)
        
        # Create stacked LSTM
        if num_layers == 1:
            self.lstm = layers.LSTM(hidden_size, return_sequences=True, return_state=True)
        else:
            # For multi-layer LSTM
            lstm_cells = [layers.LSTMCell(hidden_size) for _ in range(num_layers)]
            self.lstm = layers.RNN(lstm_cells, return_sequences=True, return_state=True)
            
        self.linear = layers.Dense(vocab_size)
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
    
    def call(self, features, captions, lengths=None, training=False):
        """
        Decode image feature vectors and generates captions.
        
        Args:
            features: image features, (batch_size, embed_size)
            captions: encoded captions, (batch_size, max_caption_length)
            lengths: valid lengths for each caption
            training: whether the call is in training mode
            
        Returns:
            outputs: predicted scores for each word in vocabulary
        """
        # Embed word indices to vectors
        embeddings = self.embed(captions)
        
        # Prepare features for input to LSTM
        features_expanded = tf.expand_dims(features, 1)
        
        # Concatenate features with embeddings
        # In TF, we need to handle padding differently than PyTorch's pack_padded_sequence
        inputs = tf.concat([features_expanded, embeddings[:, :-1]], axis=1)
        
        # Pass through LSTM
        hidden_states, *_ = self.lstm(inputs, training=training)
        
        # Predict next words
        outputs = self.linear(hidden_states)
        
        # If lengths are provided, we should mask the outputs
        if lengths is not None:
            # Create a mask based on lengths
            mask = tf.sequence_mask(lengths, tf.shape(outputs)[1], dtype=tf.float32)
            # Apply mask
            outputs = outputs * tf.expand_dims(mask, -1)
        
        return outputs
    
    def sample(self, features):
        """
        Generate captions for given image features using greedy search.
        
        Args:
            features: image features, (batch_size, embed_size)
            
        Returns:
            sampled_ids: sampled word ids, (batch_size, max_seq_length)
        """
        batch_size = tf.shape(features)[0]
        sampled_ids = []
        
        # Initial state and input
        states = None
        inputs = tf.expand_dims(features, 1)
        
        for i in range(self.max_seq_length):
            # First time step or not
            if states is None:
                outputs, h, c = self.lstm(inputs, training=False)
                states = [h, c]
            else:
                outputs, h, c = self.lstm(inputs, initial_state=states, training=False)
                states = [h, c]
            
            # Predict the next word
            outputs = self.linear(outputs)
            predicted = tf.argmax(outputs, axis=2)
            
            # Save the predicted word id
            sampled_ids.append(predicted[:, 0])
            
            # Prepare input for the next step
            inputs = self.embed(predicted)
        
        # Stack the sampled ids to a tensor and return
        sampled_ids = tf.stack(sampled_ids, axis=1)
        return sampled_ids


# Function to create the complete image captioning model
def create_image_captioning_model(embed_size, hidden_size, vocab_size, num_layers=1, max_seq_length=20):
    """
    Create the complete image captioning model with encoder and decoder.
    
    Args:
        embed_size: dimension of word embeddings
        hidden_size: dimension of LSTM hidden states
        vocab_size: size of vocabulary
        num_layers: number of LSTM layers
        max_seq_length: maximum sequence length for generation
        
    Returns:
        encoder: EncoderCNN model
        decoder: DecoderRNN model
    """
    # Create encoder and decoder
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, max_seq_length)
    
    # Build models by calling them with sample data
    sample_image = tf.random.normal((1, 224, 224, 3))
    sample_caption = tf.zeros((1, 10), dtype=tf.int32)
    
    features = encoder(sample_image)
    _ = decoder(features, sample_caption)
    
    return encoder, decoder


# Function to convert PyTorch model weights to Keras
def convert_pytorch_to_keras(pytorch_encoder_path, pytorch_decoder_path, keras_encoder, keras_decoder):
    """
    Convert PyTorch model weights to Keras format.
    
    This is a placeholder function - actual implementation would require PyTorch
    to be installed and would need to handle the specific weight mapping between
    architectures.
    
    Args:
        pytorch_encoder_path: path to the PyTorch encoder weights
        pytorch_decoder_path: path to the PyTorch decoder weights
        keras_encoder: Keras encoder model
        keras_decoder: Keras decoder model
        
    Returns:
        keras_encoder: Keras encoder with loaded weights
        keras_decoder: Keras decoder with loaded weights
    """
    import torch  
    
    # Load PyTorch model weights
    encoder_state_dict = torch.load(pytorch_encoder_path, map_location=torch.device('cpu'))
    decoder_state_dict = torch.load(pytorch_decoder_path, map_location=torch.device('cpu'))
    
    # Convert encoder weights
    # ResNet weights are already loaded from ImageNet
    
    # Convert linear layer weights
    keras_encoder.linear.set_weights([
        encoder_state_dict['linear.weight'].numpy().T,  # Transpose for Keras format
        encoder_state_dict['linear.bias'].numpy()
    ])
    
    # Convert batch norm weights
    keras_encoder.bn.set_weights([
        encoder_state_dict['bn.weight'].numpy(),        # gamma
        encoder_state_dict['bn.bias'].numpy(),          # beta
        encoder_state_dict['bn.running_mean'].numpy(),  # running mean
        encoder_state_dict['bn.running_var'].numpy()    # running variance
    ])
    
    # Convert decoder weights
    # Embedding layer
    keras_decoder.embed.set_weights([
        decoder_state_dict['embed.weight'].numpy()
    ])
    
    # LSTM layer - this is tricky due to different LSTM implementations
    # This is a simplified version and might need adjustments
    if keras_decoder.num_layers == 1:
        # For single layer LSTM
        # Extract PyTorch weights
        w_ih = decoder_state_dict['lstm.weight_ih_l0'].numpy()
        w_hh = decoder_state_dict['lstm.weight_hh_l0'].numpy()
        b_ih = decoder_state_dict['lstm.bias_ih_l0'].numpy()
        b_hh = decoder_state_dict['lstm.bias_hh_l0'].numpy()
        
        # PyTorch LSTM weights are [ifgo] format while Keras uses [iofc]
        # Reorder and concatenate weights for Keras format
        w_i, w_f, w_g, w_o = np.split(w_ih, 4)
        w_hi, w_hf, w_hg, w_ho = np.split(w_hh, 4)
        
        b_i, b_f, b_g, b_o = np.split(b_ih, 4)
        b_hi, b_hf, b_hg, b_ho = np.split(b_hh, 4)
        
        # Reorder to [i, o, f, g] for Keras
        # Concat input and recurrent weights
        keras_w = np.concatenate([
            np.concatenate([w_i, w_o, w_f, w_g], axis=0),
            np.concatenate([w_hi, w_ho, w_hf, w_hg], axis=0)
        ], axis=1)
        
        # Concat input and recurrent biases
        keras_b = np.concatenate([
            b_i + b_hi,
            b_o + b_ho,
            b_f + b_hf,
            b_g + b_hg
        ])
        
        keras_decoder.lstm.set_weights([keras_w, keras_b])
    
    # Linear layer
    keras_decoder.linear.set_weights([
        decoder_state_dict['linear.weight'].numpy().T,
        decoder_state_dict['linear.bias'].numpy()
    ])
    
    return keras_encoder, keras_decoder


# Sample usage
if __name__ == "__main__":
    # Create models
    embed_size = 256
    hidden_size = 512
    vocab_size = 10000  # Example size
    
    encoder, decoder = create_image_captioning_model(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size
    )
    
    # Print model summaries
    print("Encoder Summary:")
    encoder.summary()
    
    print("\nDecoder Summary:")
    decoder.summary()
    
    # Example of converting weights (would require PyTorch)
    # convert_pytorch_to_keras(
    #     pytorch_encoder_path="models/encoder-5-3000.pkl",
    #     pytorch_decoder_path="models/decoder-5-3000.pkl",
    #     keras_encoder=encoder,
    #     keras_decoder=decoder
    # )
    
    # Example of inference
    print("\nExample Inference:")
    sample_image = tf.random.normal((1, 224, 224, 3))
    features = encoder(sample_image, training=False)
    print(f"Feature shape: {features.shape}")
    
    sample_ids = decoder.sample(features)
    print(f"Generated caption ids shape: {sample_ids.shape}")